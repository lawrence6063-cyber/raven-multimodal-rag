"""QueryRouter — retrieval routing decision (OPTIMIZATION_SPEC §3.1).

Decides (a) whether a query needs knowledge-base retrieval at all (greetings /
general knowledge can be answered directly) and (b) which collection(s) are
relevant. LLM-proposed collections are validated against a whitelist of
available collections to prevent hallucinated / unauthorized targets.

The router is a best-effort optimization: any LLM or parsing failure degrades
conservatively to "retrieve from all collections" so nothing is ever missed.
It therefore never raises.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

from src.core.agent.agent_types import RouteDecision
from src.core.agent.json_utils import extract_json_object
from src.libs.llm.base_llm import ChatMessage
from src.observability.logger import get_logger

if TYPE_CHECKING:
    from src.core.settings import Settings
    from src.core.trace.trace_context import TraceContext
    from src.libs.llm.base_llm import BaseLLM

logger = get_logger("core.agent.router")

# _PROMPT_PATH routing prompt template path
_PROMPT_PATH = "config/prompts/agent_route.txt"

# _FALLBACK_PROMPT used when the template file is missing
_FALLBACK_PROMPT = (
    "Decide if the question needs knowledge-base retrieval and which collections "
    "apply. Choose collections only from AVAILABLE; empty array means all.\n\n"
    "Question: {query}\nAVAILABLE: {collections}\n\n"
    'Output JSON: {{"need_retrieval": true, "collections": [], "reasoning": "", '
    '"direct_answer": ""}}'
)


class QueryRouter:
    """Routes a query: whether to retrieve and which collections to target."""

    def __init__(self, settings: "Settings", llm: "BaseLLM | None" = None):
        self._settings = settings
        self._llm = llm
        self._prompt_template = self._load_prompt()

    def decide(
        self,
        query: str,
        available_collections: list[str],
        trace: "TraceContext | None" = None,
    ) -> RouteDecision:
        """Decide retrieval need and target collections for ``query``.

        Never raises: on any failure returns the conservative decision
        ``need_retrieval=True, target_collections=[]`` (retrieve all).

        Args:
            query: The user question.
            available_collections: Whitelist of valid collection names.
            trace: Optional TraceContext for the ``agent_route`` stage.

        Returns:
            A validated RouteDecision.
        """
        start = time.perf_counter()
        try:
            decision = self._decide(query, available_collections)
        except Exception as e:  # best-effort: never block retrieval on routing
            logger.warning(f"Routing failed, defaulting to retrieve-all: {e}")
            decision = RouteDecision(
                need_retrieval=True, target_collections=[], reasoning="route_degraded"
            )
        self._record(trace, start, decision)
        return decision

    def _decide(
        self, query: str, available_collections: list[str]
    ) -> RouteDecision:
        """Run the LLM and parse/validate its routing decision."""
        collections_text = ", ".join(available_collections) if available_collections else "(none)"
        prompt = self._prompt_template.format(query=query, collections=collections_text)

        llm = self._get_llm()
        response = llm.chat([ChatMessage(role="user", content=prompt)])
        data = extract_json_object(response.content)
        if data is None:
            # Conservative: parse failure means retrieve everything.
            return RouteDecision(
                need_retrieval=True, target_collections=[], reasoning="parse_fallback"
            )

        need_retrieval = bool(data.get("need_retrieval", True))
        reasoning = str(data.get("reasoning", ""))
        direct_answer = str(data.get("direct_answer", ""))

        # Whitelist validation: drop any collection not in the available set.
        proposed = data.get("collections", [])
        targets: list[str] = []
        if isinstance(proposed, list):
            available_set = set(available_collections)
            for name in proposed:
                if isinstance(name, str) and name in available_set and name not in targets:
                    targets.append(name)

        return RouteDecision(
            need_retrieval=need_retrieval,
            target_collections=targets,
            reasoning=reasoning,
            direct_answer=direct_answer,
        )

    @staticmethod
    def _record(
        trace: "TraceContext | None", start: float, decision: RouteDecision
    ) -> None:
        """Record the ``agent_route`` trace stage when a trace is present."""
        if trace is None:
            return
        elapsed = (time.perf_counter() - start) * 1000.0
        trace.record_stage(
            "agent_route",
            method="router",
            elapsed_ms=elapsed,
            need_retrieval=decision.need_retrieval,
            collections=list(decision.target_collections),
            reasoning=decision.reasoning[:200],
        )

    def _get_llm(self) -> "BaseLLM":
        """Lazily build the text LLM from settings when not injected."""
        if self._llm is None:
            from src.libs.llm.llm_factory import LLMFactory

            self._llm = LLMFactory.create(self._settings.llm)
        return self._llm

    def _load_prompt(self) -> str:
        """Load the routing prompt template, falling back to a built-in."""
        path = Path(_PROMPT_PATH)
        if path.exists():
            return path.read_text(encoding="utf-8")
        return _FALLBACK_PROMPT
