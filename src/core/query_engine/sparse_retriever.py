"""SparseRetriever — BM25 keyword-based retrieval."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.core.types import RetrievalResult
from src.ingestion.storage.bm25_indexer import BM25Indexer

if TYPE_CHECKING:
    from src.core.settings import Settings


class SparseRetriever:
    """Retrieves chunks using BM25 keyword matching.

    职责单一：仅通过 BM25Indexer 进行关键词评分，返回 chunk_id + score。
    不负责获取 chunk 全文，全文补全由上游 HybridSearch 统一处理。
    """

    def __init__(
        self,
        settings: "Settings",
        bm25_indexer: BM25Indexer | None = None,
    ):
        self._bm25 = bm25_indexer or BM25Indexer(settings.ingestion.bm25_index_path)

    def retrieve(
        self,
        keywords: list[str],
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        """Retrieve top-k chunks by BM25 keyword matching.

        Args:
            keywords: List of query keywords (from QueryProcessor).
            top_k: Number of results to return.

        Returns:
            List of RetrievalResult sorted by BM25 score descending.
            注意：text 字段为空字符串，需由上游补全。
        """
        if not keywords:
            return []

        # Query BM25 index
        bm25_results = self._bm25.query(keywords, top_k=top_k)
        if not bm25_results:
            return []

        return [
            RetrievalResult(
                chunk_id=r["chunk_id"],
                score=r["score"],
                text="",
                metadata={},
            )
            for r in bm25_results
        ]