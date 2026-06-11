"""Tests for LLM Reranker (mock LLM)."""

import pytest
from unittest.mock import patch, MagicMock

from src.libs.reranker.base_reranker import RerankCandidate, RerankerError
from src.libs.reranker.reranker_factory import RerankerFactory, _RERANKER_REGISTRY
from src.core.settings import RerankSettings

# Import to trigger registration
from src.libs.reranker.llm_reranker import LLMReranker


class TestLLMReranker:
    """Test LLM Reranker with mocked LLM."""

    def test_factory_creates_llm_reranker(self):
        settings = RerankSettings(enabled=True, provider="llm", top_n=3)
        reranker = RerankerFactory.create(settings)
        assert isinstance(reranker, LLMReranker)
        assert reranker.provider_name == "llm"

    def test_rerank_empty_candidates(self):
        settings = RerankSettings(enabled=True, provider="llm", top_n=3)
        reranker = LLMReranker(settings)
        result = reranker.rerank("query", [])
        assert result == []

    def test_parse_scores_valid_json(self):
        settings = RerankSettings(enabled=True, provider="llm", top_n=5)
        reranker = LLMReranker(settings)
        response = '[{"index": 1, "score": 9}, {"index": 0, "score": 7}]'
        scores = reranker._parse_scores(response, 2)
        assert scores == [(1, 9.0), (0, 7.0)]

    def test_parse_scores_invalid_json_fallback(self):
        settings = RerankSettings(enabled=True, provider="llm", top_n=5)
        reranker = LLMReranker(settings)
        response = "not valid json at all"
        scores = reranker._parse_scores(response, 3)
        assert len(scores) == 3
        assert all(s[1] == 5.0 for s in scores)

    def test_registered(self):
        assert "llm" in _RERANKER_REGISTRY
