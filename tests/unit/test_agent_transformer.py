"""Tests for QueryTransformer — decomposition, dedup, cap, conservative fallback."""

from __future__ import annotations

from src.core.agent.query_transformer import QueryTransformer
from src.core.settings import Settings
from src.core.trace.trace_context import TraceContext
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


def _transformer(text: str = "", error: Exception | None = None) -> QueryTransformer:
    return QueryTransformer(Settings(), llm=FakeLLM(text, error))


class TestQueryTransformer:
    def test_decomposes_into_subqueries(self):
        text = '[{"text": "what is rag", "purpose": "def"}, {"text": "rag vs fine-tuning"}]'
        subs = _transformer(text).transform("rag overview and comparison")
        assert [s.text for s in subs] == ["what is rag", "rag vs fine-tuning"]
        assert subs[0].purpose == "def"

    def test_deduplicates_normalized(self):
        text = '[{"text": "What is RAG"}, {"text": "what is rag"}, {"text": "other"}]'
        subs = _transformer(text).transform("q")
        assert [s.text for s in subs] == ["What is RAG", "other"]

    def test_caps_at_max_subqueries(self):
        s = Settings()
        s.agent.max_subqueries = 2
        t = QueryTransformer(s, llm=FakeLLM('[{"text":"a"},{"text":"b"},{"text":"c"}]'))
        subs = t.transform("q")
        assert len(subs) == 2

    def test_accepts_bare_string_items(self):
        subs = _transformer('["alpha", "beta"]').transform("q")
        assert [s.text for s in subs] == ["alpha", "beta"]

    def test_parse_failure_returns_original(self):
        subs = _transformer("garbage not json").transform("orig query")
        assert [s.text for s in subs] == ["orig query"]

    def test_empty_array_returns_original(self):
        subs = _transformer("[]").transform("orig query")
        assert [s.text for s in subs] == ["orig query"]

    def test_llm_error_returns_original(self):
        subs = _transformer(error=LLMError("x", provider="fake")).transform("orig")
        assert [s.text for s in subs] == ["orig"]

    def test_records_trace_stage(self):
        trace = TraceContext("query")
        _transformer('[{"text":"a"}]').transform("q", trace=trace)
        assert "agent_rewrite" in [s["name"] for s in trace.stages]
