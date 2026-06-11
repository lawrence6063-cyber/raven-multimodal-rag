"""HybridSearch — orchestrates Dense + Sparse + Fusion retrieval."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from src.core.types import RetrievalResult
from src.core.query_engine.query_processor import QueryProcessor
from src.core.query_engine.dense_retriever import DenseRetriever
from src.core.query_engine.sparse_retriever import SparseRetriever
from src.core.query_engine.fusion import RRFFusion
from src.libs.vector_store.vector_store_factory import VectorStoreFactory
from src.observability.logger import get_logger

if TYPE_CHECKING:
    from src.core.settings import Settings
    from src.core.trace.trace_context import TraceContext
    from src.libs.vector_store.base_vector_store import BaseVectorStore

logger = get_logger("query_engine.hybrid_search")

class HybridSearch:
    """Orchestrates hybrid retrieval: Dense + Sparse + RRF Fusion."""

    def __init__(
        self,
        settings: "Settings",
        query_processor: QueryProcessor | None = None,
        dense_retriever: DenseRetriever | None = None,
        sparse_retriever: SparseRetriever | None = None,
        fusion: RRFFusion | None = None,
        vector_store: "BaseVectorStore | None" = None,
    ):
        self._settings = settings
        self._processor = query_processor or QueryProcessor()
        self._dense = dense_retriever or DenseRetriever(settings)
        self._sparse = sparse_retriever or SparseRetriever(settings)
        self._fusion = fusion or RRFFusion(k=settings.retrieval.rrf_k)
        self._store = vector_store or VectorStoreFactory.create(settings.vector_store)
        self._top_k = settings.retrieval.top_k

    def search(
        self,
        query: str = "",
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
        trace: "TraceContext | None" = None,
        image: str | bytes | None = None,
    ) -> list[RetrievalResult]:
        """Execute hybrid search: Dense + Sparse + RRF Fusion.

        Supports three query modes for cross-modal retrieval (path B):
        - text only: Dense(text) + Sparse(keywords) + RRF (original behavior).
        - image only: Dense(image vector) only — sparse/BM25 is meaningless on an
          image, so it is skipped.
        - text + image: Dense uses the image vector (visual intent) while the text
          still drives Sparse keywords; results are RRF-fused.

        Args:
            query: User's search query text (optional when ``image`` is given).
            top_k: Number of results (default from settings).
            filters: Optional metadata filters.
            trace: Optional TraceContext for per-stage instrumentation.
            image: Optional query image (local path / bytes / base64 data URI).

        Returns:
            Fused and ranked RetrievalResult list.
        """
        k = top_k or self._top_k

        # Preprocess query
        start = time.perf_counter()
        processed = self._processor.process(query, filters)
        merged_filters = processed.filters
        self._trace_stage(
            trace, "query_processing", start, method="keyword+filter",
            keywords=len(processed.keywords),
        )

        # Dense retrieval — image vector takes precedence (cross-modal intent)
        dense_results = []
        start = time.perf_counter()
        try:
            if image is not None:
                image_vector = self._dense.embed_image_query(image)
                if image_vector:
                    dense_results = self._dense.retrieve_by_vector(
                        image_vector, top_k=k, filters=merged_filters or None
                    )
                else:
                    logger.warning("Image embedding unavailable; image query ignored")
            elif query:
                dense_results = self._dense.retrieve(query, top_k=k, filters=merged_filters or None)
            logger.info(f"Dense: {len(dense_results)} results")
        except Exception as e:
            logger.warning(f"Dense retrieval failed, continuing with sparse only: {e}")
        self._trace_stage(
            trace, "dense_retrieval", start,
            method="image_vector" if image is not None else "embedding",
            results=len(dense_results),
        )

        # Sparse retrieval
        sparse_results = []
        start = time.perf_counter()
        try:
            if processed.keywords:
                sparse_results = self._sparse.retrieve(processed.keywords, top_k=k)
                logger.info(f"Sparse: {len(sparse_results)} results")
        except Exception as e:
            logger.warning(f"Sparse retrieval failed, continuing with dense only: {e}")
        self._trace_stage(
            trace, "sparse_retrieval", start, method="bm25",
            results=len(sparse_results),
        )

        # Fusion
        start = time.perf_counter()
        if dense_results and sparse_results:
            fused = self._fusion.fuse(dense_results, sparse_results, top_k=k)
            logger.info(f"Fused: {len(fused)} results")
            fusion_method = "rrf"
        elif dense_results:
            fused = dense_results[:k]
            fusion_method = "dense_only"
        elif sparse_results:
            fused = sparse_results[:k]
            fusion_method = "sparse_only"
        else:
            fused = []
            fusion_method = "empty"
        self._trace_stage(
            trace, "fusion", start, method=fusion_method, results=len(fused),
        )

        # 补全缺失的 text（Sparse 结果只有 id+score，需要从向量库获取全文）
        fused = self._fill_missing_text(fused)

        # Post-filter by metadata if needed
        if merged_filters:
            fused = self._apply_metadata_filters(fused, merged_filters)

        return fused[:k]

    @staticmethod
    def _trace_stage(
        trace: "TraceContext | None",
        name: str,
        start: float,
        method: str = "",
        **details: Any,
    ) -> None:
        """Record a stage on the trace (if any) using elapsed time since start."""
        if trace is None:
            return
        elapsed = (time.perf_counter() - start) * 1000.0
        trace.record_stage(name, method=method, elapsed_ms=elapsed, **details)

    def _fill_missing_text(self, results: list[RetrievalResult]) -> list[RetrievalResult]:
        """补全 text 为空的检索结果。

        从向量库批量获取缺失的 chunk 全文和元数据，填充回对应的 RetrievalResult。
        如果所有结果都已有 text，则跳过 IO 操作。
        如果获取失败，记录警告日志并返回原结果（text 保持为空）。
        """
        if not results:
            return results

        # 筛选出 text 为空的结果
        missing_ids = [r.chunk_id for r in results if not r.text]
        if not missing_ids:
            return results

        try:
            records = self._store.get_by_ids(missing_ids)
            # 构建 id -> record 映射
            record_map = {rec.id: rec for rec in records}

            # 填充缺失的 text 和 metadata
            for r in results:
                if not r.text and r.chunk_id in record_map:
                    rec = record_map[r.chunk_id]
                    r.text = rec.text
                    if not r.metadata and rec.metadata:
                        r.metadata = rec.metadata

            logger.info(f"Filled text for {len(record_map)}/{len(missing_ids)} chunks")
        except Exception as e:
            logger.warning(f"Failed to fill missing text, continuing without: {e}")

        return results

    def _apply_metadata_filters(
        self, candidates: list[RetrievalResult], filters: dict[str, Any]
    ) -> list[RetrievalResult]:
        """Post-filter candidates by metadata (fallback for stores that don't support filters)."""
        filtered = []
        for r in candidates:
            match = True
            for key, value in filters.items():
                if key in r.metadata and r.metadata[key] != value:
                    match = False
                    break
            if match:
                filtered.append(r)
        return filtered
