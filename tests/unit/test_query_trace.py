"""Unit tests for Query-chain tracing (F3)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from src.core.query_engine.hybrid_search import HybridSearch
from src.core.trace.trace_context import TraceContext
from src.core.types import RetrievalResult


def _make_hybrid():
    settings = MagicMock()
    settings.retrieval.rrf_k = 60
    settings.retrieval.top_k = 5

    processor = MagicMock()
    processor.process.return_value = SimpleNamespace(keywords=["python"], filters=None)

    dense = MagicMock()
    dense.retrieve.return_value = [
        RetrievalResult(chunk_id="c1", score=0.9, text="dense text", metadata={})
    ]
    sparse = MagicMock()
    sparse.retrieve.return_value = [
        RetrievalResult(chunk_id="c2", score=0.8, text="sparse text", metadata={})
    ]
    fusion = MagicMock()
    fusion.fuse.return_value = [
        RetrievalResult(chunk_id="c1", score=0.03, text="dense text", metadata={}),
        RetrievalResult(chunk_id="c2", score=0.02, text="sparse text", metadata={}),
    ]
    store = MagicMock()
    store.get_by_ids.return_value = []

    return HybridSearch(
        settings=settings,
        query_processor=processor,
        dense_retriever=dense,
        sparse_retriever=sparse,
        fusion=fusion,
        vector_store=store,
    )


def test_search_records_query_stages():
    hs = _make_hybrid()
    trace = TraceContext(trace_type="query")

    hs.search("python programming", trace=trace)

    stage_names = [s["name"] for s in trace.stages]
    assert stage_names == [
        "query_processing",
        "dense_retrieval",
        "sparse_retrieval",
        "fusion",
    ]
    for stage in trace.stages:
        assert "elapsed_ms" in stage
        assert "method" in stage
    fusion_stage = next(s for s in trace.stages if s["name"] == "fusion")
    assert fusion_stage["method"] == "rrf"
    assert trace.trace_type == "query"


def test_search_without_trace_still_works():
    hs = _make_hybrid()
    results = hs.search("python")  # trace defaults to None
    assert len(results) == 2


def test_reranker_records_rerank_stage():
    from src.core.query_engine.reranker import QueryReranker

    settings = MagicMock()
    settings.rerank.provider = "cross_encoder"

    inner = MagicMock()
    inner.rerank.return_value = []  # RerankCandidate list
    reranker = QueryReranker.__new__(QueryReranker)
    reranker._settings = settings
    reranker._reranker = inner

    trace = TraceContext(trace_type="query")
    results = [RetrievalResult(chunk_id="c1", score=0.5, text="t", metadata={})]
    reranker.rerank("q", results, trace=trace)

    rerank_stage = next(s for s in trace.stages if s["name"] == "rerank")
    assert rerank_stage["method"] == "cross_encoder"
    assert "elapsed_ms" in rerank_stage
    assert rerank_stage["fallback"] is False
