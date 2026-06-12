"""Tests for AnswerSynthesizer — grounded answer composition with citations."""

from __future__ import annotations

import pytest

from src.core.agent.answer_synthesizer import AnswerSynthesizer
from src.core.agent.context_utils import format_numbered_context
from src.core.settings import Settings
from src.core.trace.trace_context import TraceContext
from src.core.types import RetrievalResult
from src.libs.llm.base_llm import BaseLLM, ChatResponse, LLMError


class FakeLLM(BaseLLM):
    """Returns a preset response or raises a preset error (offline)."""

    def __init__(self, response_text: str = "", error: Exception | None = None):
        self._text = response_text
        self._error = error

    def chat(self, messages, **kwargs) -> ChatResponse:
        if self._error is not None:
            raise self._error
        return ChatResponse(content=self._text)

    @property
    def provider_name(self) -> str:
        return "fake"


def _context(n: int = 2) -> list[RetrievalResult]:
    return [
        RetrievalResult(chunk_id=f"c{i}", score=1.0, text=f"passage {i} content")
        for i in range(1, n + 1)
    ]


def _synth(text: str = "", error: Exception | None = None) -> AnswerSynthesizer:
    return AnswerSynthesizer(Settings(), llm=FakeLLM(text, error))


class TestAnswerSynthesizer:
    def test_valid_answer_with_citations(self):
        text = '{"answer": "RAG combines retrieval [1] and generation [2].", "citations": [1, 2]}'
        result = _synth(text).answer("what is rag?", _context(2))
        assert "RAG combines" in result.answer
        assert result.used_citation_ids == [1, 2]

    def test_citation_out_of_range_filtered(self):
        text = '{"answer": "ans [1]", "citations": [1, 5, 0]}'
        result = _synth(text).answer("q", _context(2))
        assert result.used_citation_ids == [1]

    def test_unparseable_falls_back_to_raw_answer(self):
        result = _synth("Just a plain answer with no JSON").answer("q", _context(2))
        assert result.answer == "Just a plain answer with no JSON"
        # all passages assumed used so citations are not dropped
        assert result.used_citation_ids == [1, 2]

    def test_llm_error_propagates(self):
        with pytest.raises(LLMError):
            _synth(error=LLMError("boom", provider="fake")).answer("q", _context(1))

    def test_records_trace_stage(self):
        trace = TraceContext("query")
        _synth('{"answer": "a", "citations": [1]}').answer("q", _context(1), trace=trace)
        names = [s["name"] for s in trace.stages]
        assert "agent_synthesize" in names


class TestFormatNumberedContext:
    def test_numbering_is_one_based(self):
        out = format_numbered_context(_context(2))
        assert out.startswith("[1] passage 1")
        assert "[2] passage 2" in out

    def test_truncation(self):
        long = [RetrievalResult(chunk_id="c", score=1.0, text="x" * 500)]
        out = format_numbered_context(long, max_chars=10)
        assert out == "[1] " + "x" * 10

    def test_empty_context(self):
        assert format_numbered_context([]) == ""
