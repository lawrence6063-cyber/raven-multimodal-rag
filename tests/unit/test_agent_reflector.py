"""Tests for Reflector — sufficiency verdict, follow-ups, conservative fallback."""

from __future__ import annotations

from src.core.agent.reflector import Reflector
from src.core.settings import Settings
from src.core.trace.trace_context import TraceContext
from src.core.types import RetrievalResult
from src.libs.llm.base_llm import BaseLLM, ChatResponse, LLMError


class FakeLLM(BaseLLM):
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


def _ctx(n=1):
    return [RetrievalResult(chunk_id=f"c{i}", score=1.0, text=f"p{i}") for i in range(1, n + 1)]


def _reflector(text: str = "", error: Exception | None = None) -> Reflector:
    return Reflector(Settings(), llm=FakeLLM(text, error))


class TestReflector:
    def test_sufficient_verdict(self):
        v = _reflector('{"sufficient": true, "follow_up": [], "reasoning": "ok"}').assess(
            "q", _ctx(2)
        )
        assert v.sufficient is True
        assert v.follow_up_queries == []

    def test_insufficient_with_followups(self):
        text = '{"sufficient": false, "follow_up": ["q1", "q2"], "reasoning": "gap"}'
        v = _reflector(text).assess("q", _ctx(1))
        assert v.sufficient is False
        assert v.follow_up_queries == ["q1", "q2"]

    def test_insufficient_without_followups_treated_sufficient(self):
        # No actionable follow-up → nothing to do → stop the loop.
        v = _reflector('{"sufficient": false, "follow_up": []}').assess("q", _ctx(1))
        assert v.sufficient is True

    def test_parse_failure_is_sufficient(self):
        v = _reflector("not json").assess("q", _ctx(1))
        assert v.sufficient is True
        assert v.reasoning == "parse_fallback"

    def test_llm_error_is_sufficient(self):
        v = _reflector(error=LLMError("x", provider="fake")).assess("q", _ctx(1))
        assert v.sufficient is True
        assert v.reasoning == "reflect_degraded"

    def test_records_trace_stage(self):
        trace = TraceContext("query")
        _reflector('{"sufficient": true}').assess("q", _ctx(1), trace=trace)
        assert "agent_reflect" in [s["name"] for s in trace.stages]
