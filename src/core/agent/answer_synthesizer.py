"""AnswerSynthesizer — server-side LLM answer composition with citations.

Given the user query and the accumulated retrieval context, the synthesizer
asks the LLM to compose an answer grounded ONLY in the numbered passages and to
report which passages it cited. This is the server-side answer layer that the
project previously lacked (answers were composed by the external MCP client);
it is required for self-correction (the reflector reuses the same grounding).

The LLM is injectable for offline testing; when omitted it is lazily built from
``settings.llm`` via ``LLMFactory``. JSON parsing is robust (best-effort with a
conservative fallback); upstream LLM/API errors propagate so the orchestrator
can apply its global degradation path.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

from src.core.agent.agent_types import SynthResult
from src.core.agent.context_utils import format_numbered_context
from src.core.agent.json_utils import extract_json_object
from src.core.types import RetrievalResult
from src.libs.llm.base_llm import ChatMessage
from src.observability.logger import get_logger

if TYPE_CHECKING:
    from src.core.settings import Settings
    from src.core.trace.trace_context import TraceContext
    from src.libs.llm.base_llm import BaseLLM

logger = get_logger("core.agent.answer_synthesizer")

# _PROMPT_PATH answer-synthesis prompt template path
_PROMPT_PATH = "config/prompts/agent_answer.txt"

# _FALLBACK_PROMPT used when the template file is missing
_FALLBACK_PROMPT = (
    "Answer the question using ONLY the numbered passages. Cite used passages "
    'as [n]. If insufficient, say so.\n\nQuestion: {query}\n\nPassages:\n'
    '{context}\n\nOutput JSON: {{"answer": "...", "citations": [1]}}'
)


class AnswerSynthesizer:
    """Composes a cited answer from retrieval context using the text LLM."""

    def __init__(self, settings: "Settings", llm: "BaseLLM | None" = None):
        self._settings = settings
        self._llm = llm
        self._prompt_template = self._load_prompt()

    def answer(
        self,
        query: str,
        context: list[RetrievalResult],
        trace: "TraceContext | None" = None,
    ) -> SynthResult:
        """Compose an answer grounded in ``context``.

        Args:
            query: The original user question.
            context: Numbered retrieval passages (1-based citation order).
            trace: Optional TraceContext for the ``agent_synthesize`` stage.

        Returns:
            SynthResult with the answer text and the cited passage indices.

        Raises:
            LLMError: If the underlying LLM call fails (propagated for the
                orchestrator's global fallback).
        """
        start = time.perf_counter()
        context_text = format_numbered_context(context)
        prompt = self._prompt_template.format(query=query, context=context_text)

        llm = self._get_llm()
        response = llm.chat([ChatMessage(role="user", content=prompt)])
        result = self._parse(response.content, len(context))

        self._record(trace, start, result)
        return result

    def _parse(self, response_text: str, n_context: int) -> SynthResult:
        """Parse the LLM JSON into a SynthResult (best-effort).

        On unparseable output, treat the whole response as the answer and assume
        all passages were used so citations are not silently dropped.
        """
        data = extract_json_object(response_text)
        if data is None:
            answer = (response_text or "").strip()
            return SynthResult(
                answer=answer,
                used_citation_ids=list(range(1, n_context + 1)),
            )

        answer = str(data.get("answer", "")).strip()
        raw_citations = data.get("citations", [])
        used: list[int] = []
        if isinstance(raw_citations, list):
            for item in raw_citations:
                try:
                    idx = int(item)
                except (TypeError, ValueError):
                    continue
                if 1 <= idx <= n_context and idx not in used:
                    used.append(idx)
        return SynthResult(answer=answer, used_citation_ids=used)

    @staticmethod
    def _record(
        trace: "TraceContext | None", start: float, result: SynthResult
    ) -> None:
        """Record the ``agent_synthesize`` trace stage when a trace is present."""
        if trace is None:
            return
        elapsed = (time.perf_counter() - start) * 1000.0
        trace.record_stage(
            "agent_synthesize",
            method="synthesizer",
            elapsed_ms=elapsed,
            answer_len=len(result.answer),
            n_citations=len(result.used_citation_ids),
        )

    def _get_llm(self) -> "BaseLLM":
        """Lazily build the text LLM from settings when not injected."""
        if self._llm is None:
            from src.libs.llm.llm_factory import LLMFactory

            self._llm = LLMFactory.create(self._settings.llm)
        return self._llm

    def _load_prompt(self) -> str:
        """Load the answer prompt template, falling back to a built-in."""
        path = Path(_PROMPT_PATH)
        if path.exists():
            return path.read_text(encoding="utf-8")
        return _FALLBACK_PROMPT
