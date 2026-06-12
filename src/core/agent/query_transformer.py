"""QueryTransformer — query rewriting and decomposition (OPTIMIZATION_SPEC §3.2).

Rewrites a colloquial/compound question into one or more focused, keyword-rich
sub-queries to improve retrieval. Like the router, this is a best-effort
optimization: any LLM/parsing failure or empty result degrades to the original
query as a single sub-query, so retrieval always proceeds. It never raises.

Sub-queries are de-duplicated (normalized text) and capped at
``settings.agent.max_subqueries``.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

from src.core.agent.agent_types import SubQuery
from src.core.agent.json_utils import extract_json_array
from src.libs.llm.base_llm import ChatMessage
from src.observability.logger import get_logger

if TYPE_CHECKING:
    from src.core.settings import Settings
    from src.core.trace.trace_context import TraceContext
    from src.libs.llm.base_llm import BaseLLM

logger = get_logger("core.agent.query_transformer")

# _PROMPT_PATH rewrite/decomposition prompt template path
_PROMPT_PATH = "config/prompts/agent_rewrite.txt"

# _FALLBACK_PROMPT used when the template file is missing
_FALLBACK_PROMPT = (
    "Rewrite the question into at most {max_subqueries} focused search "
    "sub-queries. Decompose only when there are distinct needs.\n\n"
    'Question: {query}\n\nOutput JSON array: [{{"text": "...", "purpose": ""}}]'
)


class QueryTransformer:
    """Rewrites and decomposes a query into focused sub-queries."""

    def __init__(self, settings: "Settings", llm: "BaseLLM | None" = None):
        self._settings = settings
        self._llm = llm
        self._prompt_template = self._load_prompt()

    def transform(
        self, query: str, trace: "TraceContext | None" = None
    ) -> list[SubQuery]:
        """Decompose ``query`` into focused sub-queries.

        Never raises: on any failure returns ``[SubQuery(text=query)]``.

        Args:
            query: The original user question.
            trace: Optional TraceContext for the ``agent_rewrite`` stage.

        Returns:
            A de-duplicated, capped list of SubQuery (always non-empty).
        """
        start = time.perf_counter()
        try:
            subqueries = self._transform(query)
        except Exception as e:  # best-effort: never block retrieval on rewrite
            logger.warning(f"Query transform failed, using original query: {e}")
            subqueries = [SubQuery(text=query)]

        if not subqueries:
            subqueries = [SubQuery(text=query)]

        self._record(trace, start, subqueries)
        return subqueries

    def _transform(self, query: str) -> list[SubQuery]:
        """Run the LLM and parse sub-queries (capped + de-duplicated)."""
        max_subqueries = self._settings.agent.max_subqueries
        prompt = self._prompt_template.format(query=query, max_subqueries=max_subqueries)

        llm = self._get_llm()
        response = llm.chat([ChatMessage(role="user", content=prompt)])
        data = extract_json_array(response.content)
        if data is None:
            return [SubQuery(text=query)]

        seen: set[str] = set()
        subqueries: list[SubQuery] = []
        for item in data:
            text = self._extract_text(item)
            if not text:
                continue
            key = text.strip().lower()
            if key in seen:
                continue
            seen.add(key)
            purpose = item.get("purpose", "") if isinstance(item, dict) else ""
            subqueries.append(SubQuery(text=text, purpose=str(purpose)))
            if len(subqueries) >= max_subqueries:
                break
        return subqueries

    @staticmethod
    def _extract_text(item) -> str:
        """Extract sub-query text from a JSON item (object or bare string)."""
        if isinstance(item, str):
            return item.strip()
        if isinstance(item, dict):
            return str(item.get("text", "")).strip()
        return ""

    @staticmethod
    def _record(
        trace: "TraceContext | None", start: float, subqueries: list[SubQuery]
    ) -> None:
        """Record the ``agent_rewrite`` trace stage when a trace is present."""
        if trace is None:
            return
        elapsed = (time.perf_counter() - start) * 1000.0
        trace.record_stage(
            "agent_rewrite",
            method="transformer",
            elapsed_ms=elapsed,
            n_subqueries=len(subqueries),
            subqueries=[sq.text for sq in subqueries],
        )

    def _get_llm(self) -> "BaseLLM":
        """Lazily build the text LLM from settings when not injected."""
        if self._llm is None:
            from src.libs.llm.llm_factory import LLMFactory

            self._llm = LLMFactory.create(self._settings.llm)
        return self._llm

    def _load_prompt(self) -> str:
        """Load the rewrite prompt template, falling back to a built-in."""
        path = Path(_PROMPT_PATH)
        if path.exists():
            return path.read_text(encoding="utf-8")
        return _FALLBACK_PROMPT
