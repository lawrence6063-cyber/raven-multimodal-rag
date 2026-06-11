"""Cross-Encoder Reranker — uses local cross-encoder model for reranking."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.libs.reranker.base_reranker import BaseReranker, RerankCandidate, RerankerError
from src.libs.reranker.reranker_factory import register_reranker

if TYPE_CHECKING:
    from src.core.settings import RerankSettings


@register_reranker("cross_encoder")
class CrossEncoderReranker(BaseReranker):
    """Reranker using sentence-transformers CrossEncoder model."""

    def __init__(self, settings: "RerankSettings"):
        self._settings = settings
        self._model_name = settings.model or "cross-encoder/ms-marco-MiniLM-L-6-v2"
        self._top_n = settings.top_n
        self._encoder = None

    def _get_encoder(self):
        """Lazy-load the CrossEncoder model."""
        if self._encoder is None:
            try:
                from sentence_transformers import CrossEncoder
                self._encoder = CrossEncoder(self._model_name)
            except ImportError:
                raise RerankerError(
                    "sentence-transformers not installed. Run: pip install sentence-transformers",
                    provider="cross_encoder",
                )
            except Exception as e:
                raise RerankerError(f"Failed to load model '{self._model_name}': {e}", provider="cross_encoder") from e
        return self._encoder

    def rerank(self, query: str, candidates: list[RerankCandidate]) -> list[RerankCandidate]:
        """Rerank using cross-encoder scoring."""
        if not candidates:
            return []

        try:
            encoder = self._get_encoder()
            pairs = [[query, c.text] for c in candidates]
            scores = encoder.predict(pairs)

            for i, score in enumerate(scores):
                candidates[i].score = float(score)

            ranked = sorted(candidates, key=lambda c: c.score, reverse=True)
            return ranked[: self._top_n]

        except RerankerError:
            raise
        except Exception as e:
            raise RerankerError(f"Cross-encoder rerank failed: {e}", provider="cross_encoder") from e

    @property
    def provider_name(self) -> str:
        return "cross_encoder"
