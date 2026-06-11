"""Tests for QueryReranker (Core layer with fallback)."""

import pytest
from unittest.mock import MagicMock, patch

from src.core.types import RetrievalResult
from src.core.query_engine.reranker import QueryReranker
from src.core.settings import Settings, RerankSettings
from src.libs.reranker.base_reranker import RerankCandidate, RerankerError


class TestQueryReranker:
    def test_rerank_disabled_passthrough(self):
        settings = Settings()
        settings.rerank = RerankSettings(enabled=False, provider="none")
        reranker = QueryReranker(settings)

        results = [
            RetrievalResult(chunk_id="c1", score=0.9, text="first", metadata={}),
            RetrievalResult(chunk_id="c2", score=0.7, text="second", metadata={}),
        ]
        ranked = reranker.rerank("query", results)

        # NoneReranker preserves order
        assert [r.chunk_id for r in ranked] == ["c1", "c2"]

    def test_rerank_empty_input(self):
        settings = Settings()
        settings.rerank = RerankSettings(enabled=False, provider="none")
        reranker = QueryReranker(settings)
        assert reranker.rerank("q", []) == []

    @patch("src.core.query_engine.reranker.RerankerFactory")
    def test_fallback_on_error(self, mock_factory):
        mock_reranker = MagicMock()
        mock_reranker.rerank.side_effect = RerankerError("timeout", provider="llm")
        mock_factory.create.return_value = mock_reranker

        settings = Settings()
        settings.rerank = RerankSettings(enabled=True, provider="llm")
        reranker = QueryReranker(settings)

        results = [
            RetrievalResult(chunk_id="c1", score=0.9, text="t", metadata={}),
        ]
        ranked = reranker.rerank("q", results)

        # Should return original on failure
        assert len(ranked) == 1
        assert ranked[0].chunk_id == "c1"
        assert ranked[0].metadata.get("rerank_fallback") is True
