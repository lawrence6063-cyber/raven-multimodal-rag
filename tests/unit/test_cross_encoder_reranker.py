"""Tests for Cross-Encoder Reranker (mock scorer)."""

import pytest
from unittest.mock import patch, MagicMock

from src.libs.reranker.base_reranker import RerankCandidate, RerankerError
from src.libs.reranker.reranker_factory import RerankerFactory, _RERANKER_REGISTRY
from src.core.settings import RerankSettings

# Import to trigger registration
from src.libs.reranker.cross_encoder_reranker import CrossEncoderReranker


class TestCrossEncoderReranker:
    """Test Cross-Encoder Reranker with mocked scorer."""

    def test_factory_creates_cross_encoder(self):
        settings = RerankSettings(enabled=True, provider="cross_encoder", model="test-model", top_n=3)
        reranker = RerankerFactory.create(settings)
        assert isinstance(reranker, CrossEncoderReranker)
        assert reranker.provider_name == "cross_encoder"

    def test_rerank_empty_candidates(self):
        settings = RerankSettings(enabled=True, provider="cross_encoder", top_n=3)
        reranker = CrossEncoderReranker(settings)
        result = reranker.rerank("query", [])
        assert result == []

    @patch("src.libs.reranker.cross_encoder_reranker.CrossEncoderReranker._get_encoder")
    def test_rerank_with_mock_scorer(self, mock_get_encoder):
        mock_encoder = MagicMock()
        mock_encoder.predict.return_value = [0.3, 0.9, 0.1]  # scores for 3 candidates
        mock_get_encoder.return_value = mock_encoder

        settings = RerankSettings(enabled=True, provider="cross_encoder", top_n=2)
        reranker = CrossEncoderReranker(settings)

        candidates = [
            RerankCandidate(id="c1", text="first"),
            RerankCandidate(id="c2", text="second"),
            RerankCandidate(id="c3", text="third"),
        ]
        result = reranker.rerank("query", candidates)

        # Should be sorted by score descending, top_n=2
        assert len(result) == 2
        assert result[0].id == "c2"  # score 0.9
        assert result[1].id == "c1"  # score 0.3

    def test_registered(self):
        assert "cross_encoder" in _RERANKER_REGISTRY
