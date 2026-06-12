"""Reflector — self-correction / sufficiency assessment (OPTIMIZATION_SPEC §3.4).

Given the original query and the accumulated retrieval context, the reflector
judges whether the context is sufficient to answer (CRAG/Self-RAG style). When
insufficient it proposes follow-up sub-queries that the orchestrator feeds back
into the multi-hop loop.

Conservative by design: any LLM/parsing failure yields ``sufficient=True`` so
the loop terminates rather than spinning (cost/latency safety). It never raises.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

from src.core.agent.agent_types import ReflectVerdict
from src.core.agent.context_utils import format_numbered_context
from src.core.agent.json_utils import extract_json_object
from src.core.types import RetrievalResult
from src.libs.llm.base_llm import ChatMessage
from src.observability.logger import get_logger

if TYPE_CHECKING:
    from src.core.settings import Settings
    from src.core.trace.trace_context import TraceContext
    from src.libs.llm.base_llm import BaseLLM

logger = get_logger("core.agent.reflector")

# _PROMPT_PATH sufficiency-assessment prompt template path
_PROMPT_PATH = "config/prompts/agent_reflect.txt"

# _FALLBACK_PROMPT used when the template file is missing
_FALLBACK_PROMPT = (
    "Judge if the passages suffice to answer the question. If not, propose "
    "follow-up search queries.\n\nQuestion: {query}\n\nPassages:\n{context}\n\n"
    'Output JSON: {{"sufficient": true, "follow_up": [], "reasoning": ""}}'
)


class Reflector:
    """Assesses context sufficiency and proposes follow-up queries."""

    def __init__(self, settings: "Settings", llm: "BaseLLM | None" = None):
        self._settings = settings
        self._llm = llm
        self._prompt_template = self._load_prompt()

    def assess(
        self,
        query: str,
        context: list[RetrievalResult],
        trace: "TraceContext | None" = None,
    ) -> ReflectVerdict:
        """Assess whether ``context`` suffices to answer ``query``.

        Never raises: on any failure returns ``sufficient=True`` (terminate).

        Args:
            query: The original user question.
            context: Accumulated retrieval passages.
            trace: Optional TraceContext for the ``agent_reflect`` stage.

        Returns:
            A ReflectVerdict with sufficiency and (when insufficient) follow-ups.
        """
        start = time.perf_counter()
        try:
            verdict = self._assess(query, context)
        except Exception as e:  # conservative: stop the loop rather than spin
            logger.warning(f"Reflection failed, treating as sufficient: {e}")
            verdict = ReflectVerdict(sufficient=True, reasoning="reflect_degraded")
        self._record(trace, start, verdict)
        return verdict

    def _assess(
        self, query: str, context: list[RetrievalResult]
    ) -> ReflectVerdict:
        """Run the LLM and parse the sufficiency verdict."""
        context_text = format_numbered_context(context)
        prompt = self._prompt_template.format(query=query, context=context_text)

        llm = self._get_llm()
        response = llm.chat([ChatMessage(role="user", content=prompt)])
        data = extract_json_object(response.content)
        if data is None:
            # Conservative: unparseable verdict means stop.
            return ReflectVerdict(sufficient=True, reasoning="parse_fallback")

        sufficient = bool(data.get("sufficient", True))
        reasoning = str(data.get("reasoning", ""))
        follow_up: list[str] = []
        if not sufficient:
            raw = data.get("follow_up", [])
            if isinstance(raw, list):
                for item in raw:
                    if isinstance(item, str) and item.strip():
                        follow_up.append(item.strip())
            # No concrete follow-ups → nothing actionable, treat as sufficient.
            if not follow_up:
                sufficient = True
        return ReflectVerdict(
            sufficient=sufficient, follow_up_queries=follow_up, reasoning=reasoning
        )

    @staticmethod
    def _record(
        trace: "TraceContext | None", start: float, verdict: ReflectVerdict
    ) -> None:
        """Record the ``agent_reflect`` trace stage when a trace is present."""
        if trace is None:
            return
        elapsed = (time.perf_counter() - start) * 1000.0
        trace.record_stage(
            "agent_reflect",
            method="reflector",
            elapsed_ms=elapsed,
            sufficient=verdict.sufficient,
            n_followup=len(verdict.follow_up_queries),
        )

    def _get_llm(self) -> "BaseLLM":
        """Lazily build the text LLM from settings when not injected."""
        if self._llm is None:
            from src.libs.llm.llm_factory import LLMFactory

            self._llm = LLMFactory.create(self._settings.llm)
        return self._llm

    def _load_prompt(self) -> str:
        """Load the reflection prompt template, falling back to a built-in."""
        path = Path(_PROMPT_PATH)
        if path.exists():
            return path.read_text(encoding="utf-8")
        return _FALLBACK_PROMPT
