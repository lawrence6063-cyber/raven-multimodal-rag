"""Reranker — Core layer reranking with fallback to original order."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from src.core.types import RetrievalResult
from src.libs.reranker.base_reranker import RerankCandidate, RerankerError
from src.libs.reranker.reranker_factory import RerankerFactory
from src.observability.logger import get_logger

if TYPE_CHECKING:
    from src.core.settings import Settings
    from src.core.trace.trace_context import TraceContext

logger = get_logger("query_engine.reranker")


class QueryReranker:
    """Core-layer reranker that wraps libs.reranker with fallback logic."""

    def __init__(self, settings: "Settings"):
        self._settings = settings
        self._reranker = RerankerFactory.create(settings.rerank)

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        trace: "TraceContext | None" = None,
    ) -> list[RetrievalResult]:
        """Rerank retrieval results, with fallback on failure.

        Args:
            query: The original search query.
            results: Retrieval results from HybridSearch.
            trace: Optional TraceContext for per-stage instrumentation.

        Returns:
            Reranked results. On failure, returns original results with fallback=True marker.
        """
        if not results:
            if trace is not None:
                trace.record_stage("rerank", method="skip", elapsed_ms=0.0, results=0)
            return []

        # Convert to reranker candidates
        candidates = [
            RerankCandidate(
                id=r.chunk_id,
                text=r.text,
                score=r.score,
                metadata=r.metadata,
            )
            for r in results
        ]

        provider = getattr(self._settings.rerank, "provider", "")
        start = time.perf_counter()
        try:
            ranked = self._reranker.rerank(query, candidates)
            logger.info(f"Reranked {len(results)} -> {len(ranked)} results")

            reranked = [
                RetrievalResult(
                    chunk_id=c.id,
                    score=c.score,
                    text=c.text,
                    metadata=c.metadata,
                )
                for c in ranked
            ]
            self._trace_rerank(trace, start, provider, len(reranked), fallback=False)
            return reranked

        except RerankerError as e:
            logger.warning(f"Reranker failed, falling back to original order: {e}")
            for r in results:
                r.metadata["rerank_fallback"] = True
            self._trace_rerank(trace, start, provider, len(results), fallback=True)
            return results

        except Exception as e:
            logger.error(f"Unexpected reranker error, falling back: {e}")
            for r in results:
                r.metadata["rerank_fallback"] = True
            self._trace_rerank(trace, start, provider, len(results), fallback=True)
            return results

    @staticmethod
    def _trace_rerank(
        trace: "TraceContext | None",
        start: float,
        provider: str,
        results: int,
        fallback: bool,
    ) -> None:
        """Record the rerank stage on the trace (if any)."""
        if trace is None:
            return
        elapsed = (time.perf_counter() - start) * 1000.0
        trace.record_stage(
            "rerank", method=provider, elapsed_ms=elapsed,
            results=results, fallback=fallback,
        )
