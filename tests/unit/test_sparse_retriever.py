"""Tests for SparseRetriever — verifies BM25-only retrieval without vector store dependency."""

import pytest
from unittest.mock import MagicMock, patch

from src.core.types import RetrievalResult
from src.core.query_engine.sparse_retriever import SparseRetriever


class TestSparseRetriever:
    """验证 SparseRetriever 只依赖 BM25Indexer，不依赖向量库。"""

    def _make_retriever(self, bm25_results=None):
        """创建一个带 mock BM25Indexer 的 SparseRetriever。"""
        mock_bm25 = MagicMock()
        mock_bm25.query.return_value = bm25_results or []

        mock_settings = MagicMock()
        mock_settings.ingestion.bm25_index_path = "/tmp/test_bm25"

        return SparseRetriever(settings=mock_settings, bm25_indexer=mock_bm25), mock_bm25

    def test_retrieve_returns_id_and_score_only(self):
        """验证返回结果只包含 chunk_id 和 score，text 为空字符串。"""
        bm25_results = [
            {"chunk_id": "c1", "score": 8.5},
            {"chunk_id": "c2", "score": 5.2},
        ]
        retriever, _ = self._make_retriever(bm25_results)

        results = retriever.retrieve(["python", "machine"], top_k=10)

        assert len(results) == 2
        assert all(isinstance(r, RetrievalResult) for r in results)
        assert results[0].chunk_id == "c1"
        assert results[0].score == 8.5
        assert results[0].text == ""
        assert results[0].metadata == {}
        assert results[1].chunk_id == "c2"
        assert results[1].score == 5.2
        assert results[1].text == ""

    def test_retrieve_empty_keywords_returns_empty(self):
        """验证空关键词列表直接返回空结果。"""
        retriever, mock_bm25 = self._make_retriever()

        results = retriever.retrieve([], top_k=10)

        assert results == []
        mock_bm25.query.assert_not_called()

    def test_retrieve_no_bm25_results_returns_empty(self):
        """验证 BM25 无结果时返回空列表。"""
        retriever, _ = self._make_retriever(bm25_results=[])

        results = retriever.retrieve(["nonexistent"], top_k=5)

        assert results == []

    def test_retrieve_passes_top_k_to_bm25(self):
        """验证 top_k 参数正确传递给 BM25Indexer。"""
        retriever, mock_bm25 = self._make_retriever(bm25_results=[])

        retriever.retrieve(["test"], top_k=7)

        mock_bm25.query.assert_called_once_with(["test"], top_k=7)

    def test_no_vector_store_dependency(self):
        """验证 SparseRetriever 构造函数不需要 vector_store 参数。"""
        mock_settings = MagicMock()
        mock_settings.ingestion.bm25_index_path = "/tmp/test"
        mock_bm25 = MagicMock()

        # 不传 vector_store 参数，应该正常构造
        retriever = SparseRetriever(settings=mock_settings, bm25_indexer=mock_bm25)
        assert retriever is not None

    def test_retrieve_preserves_order(self):
        """验证返回结果保持 BM25 的排序顺序。"""
        bm25_results = [
            {"chunk_id": "top", "score": 10.0},
            {"chunk_id": "mid", "score": 5.0},
            {"chunk_id": "low", "score": 1.0},
        ]
        retriever, _ = self._make_retriever(bm25_results)

        results = retriever.retrieve(["query"], top_k=3)

        assert [r.chunk_id for r in results] == ["top", "mid", "low"]
        assert [r.score for r in results] == [10.0, 5.0, 1.0]