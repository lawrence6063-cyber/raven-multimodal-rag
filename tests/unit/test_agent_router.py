"""Tests for QueryRouter — routing decision, whitelist, conservative fallback."""

from __future__ import annotations

from src.core.agent.router import QueryRouter
from src.core.settings import Settings
from src.core.trace.trace_context import TraceContext
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


def _router(text: str = "", error: Exception | None = None) -> QueryRouter:
    return QueryRouter(Settings(), llm=FakeLLM(text, error))


class TestQueryRouter:
    def test_valid_decision_with_whitelist_filtering(self):
        # "phantom" is not available and must be dropped.
        text = (
            '{"need_retrieval": true, "collections": ["rag", "phantom"], '
            '"reasoning": "needs docs"}'
        )
        decision = _router(text).decide("what is rag?", ["rag", "llm"])
        assert decision.need_retrieval is True
        assert decision.target_collections == ["rag"]
        assert decision.reasoning == "needs docs"

    def test_no_retrieval_with_direct_answer(self):
        text = (
            '{"need_retrieval": false, "collections": [], '
            '"reasoning": "greeting", "direct_answer": "Hello!"}'
        )
        decision = _router(text).decide("hi", ["rag"])
        assert decision.need_retrieval is False
        assert decision.direct_answer == "Hello!"

    def test_parse_failure_degrades_to_retrieve_all(self):
        decision = _router("not json at all").decide("q", ["rag"])
        assert decision.need_retrieval is True
        assert decision.target_collections == []
        assert decision.reasoning == "parse_fallback"

    def test_llm_error_degrades_without_raising(self):
        decision = _router(error=LLMError("boom", provider="fake")).decide("q", ["rag"])
        assert decision.need_retrieval is True
        assert decision.target_collections == []
        assert decision.reasoning == "route_degraded"

    def test_collections_wrong_type_ignored(self):
        text = '{"need_retrieval": true, "collections": "rag"}'
        decision = _router(text).decide("q", ["rag"])
        assert decision.target_collections == []

    def test_records_trace_stage(self):
        trace = TraceContext("query")
        _router('{"need_retrieval": true, "collections": []}').decide(
            "q", ["rag"], trace=trace
        )
        names = [s["name"] for s in trace.stages]
        assert "agent_route" in names
