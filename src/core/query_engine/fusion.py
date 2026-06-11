"""RRF Fusion — Reciprocal Rank Fusion for combining multiple result lists."""

from __future__ import annotations

from collections import defaultdict

from src.core.types import RetrievalResult


class RRFFusion:
    """Fuses multiple ranked result lists using Reciprocal Rank Fusion.

    RRF score = sum(1 / (k + rank_i)) for each list where the document appears.
    """

    def __init__(self, k: int = 60):
        """Initialize RRF with constant k.

        Args:
            k: RRF constant (default 60, as per original paper).
        """
        self._k = k

    def fuse(self, *result_lists: list[RetrievalResult], top_k: int = 10) -> list[RetrievalResult]:
        """Fuse multiple result lists using RRF.

        Args:
            *result_lists: Variable number of ranked result lists.
            top_k: Number of results to return after fusion.

        Returns:
            Fused and re-ranked list of RetrievalResult.
        """
        scores: dict[str, float] = defaultdict(float)
        result_map: dict[str, RetrievalResult] = {}

        for results in result_lists:
            for rank, result in enumerate(results, start=1):
                rrf_score = 1.0 / (self._k + rank)
                scores[result.chunk_id] += rrf_score
                # Keep the result with highest original score for metadata
                if result.chunk_id not in result_map or result.score > result_map[result.chunk_id].score:
                    result_map[result.chunk_id] = result

        # Sort by fused RRF score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        return [
            RetrievalResult(
                chunk_id=chunk_id,
                score=rrf_score,
                text=result_map[chunk_id].text,
                metadata=result_map[chunk_id].metadata,
            )
            for chunk_id, rrf_score in ranked
        ]
