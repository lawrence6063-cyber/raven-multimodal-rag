"""Tests for HybridSearch._fill_missing_text — verifies text completion logic."""

import pytest
from unittest.mock import MagicMock, patch

from src.core.types import RetrievalResult
from src.core.query_engine.hybrid_search import HybridSearch
from src.libs.vector_store.base_vector_store import QueryResult, VectorStoreError


class TestHybridSearchFillMissingText:
    """验证 HybridSearch 在 Fusion 之后补全缺失 text 的逻辑。"""

    def _make_hybrid_search(self, store_results=None, store_raises=None):
        """创建一个带 mock 依赖的 HybridSearch 实例。"""
        mock_settings = MagicMock()
        mock_settings.retrieval.rrf_k = 60
        mock_settings.retrieval.top_k = 10

        mock_store = MagicMock()
        if store_raises:
            mock_store.get_by_ids.side_effect = store_raises
        else:
            mock_store.get_by_ids.return_value = store_results or []

        mock_processor = MagicMock()
        mock_dense = MagicMock()
        mock_sparse = MagicMock()
        mock_fusion = MagicMock()

        hs = HybridSearch(
            settings=mock_settings,
            query_processor=mock_processor,
            dense_retriever=mock_dense,
            sparse_retriever=mock_sparse,
            fusion=mock_fusion,
            vector_store=mock_store,
        )
        return hs, mock_store

    def test_all_results_have_text_skips_io(self):
        """所有结果都有 text 时，不调用 get_by_ids。"""
        hs, mock_store = self._make_hybrid_search()

        results = [
            RetrievalResult(chunk_id="c1", score=0.9, text="hello", metadata={}),
            RetrievalResult(chunk_id="c2", score=0.8, text="world", metadata={}),
        ]

        filled = hs._fill_missing_text(results)

        mock_store.get_by_ids.assert_not_called()
        assert filled[0].text == "hello"
        assert filled[1].text == "world"

    def test_missing_text_gets_filled(self):
        """部分结果缺 text 时，调用 get_by_ids 补全。"""
        store_results = [
            QueryResult(id="c2", score=0.0, text="filled text", metadata={"source": "doc1"}),
        ]
        hs, mock_store = self._make_hybrid_search(store_results=store_results)

        results = [
            RetrievalResult(chunk_id="c1", score=0.9, text="already has text", metadata={}),
            RetrievalResult(chunk_id="c2", score=0.8, text="", metadata={}),
        ]

        filled = hs._fill_missing_text(results)

        mock_store.get_by_ids.assert_called_once_with(["c2"])
        assert filled[0].text == "already has text"  # 未被修改
        assert filled[1].text == "filled text"  # 被补全
        assert filled[1].metadata == {"source": "doc1"}  # metadata 也被补全

    def test_get_by_ids_failure_degrades_gracefully(self):
        """get_by_ids 抛异常时，降级返回原结果（text 保持为空）。"""
        hs, mock_store = self._make_hybrid_search(
            store_raises=VectorStoreError("Connection failed", provider="chroma")
        )

        results = [
            RetrievalResult(chunk_id="c1", score=0.9, text="", metadata={}),
        ]

        filled = hs._fill_missing_text(results)

        assert filled[0].text == ""  # 保持为空，未崩溃
        assert len(filled) == 1

    def test_empty_results_returns_empty(self):
        """空结果列表直接返回。"""
        hs, mock_store = self._make_hybrid_search()

        filled = hs._fill_missing_text([])

        assert filled == []
        mock_store.get_by_ids.assert_not_called()

    def test_partial_ids_found_in_store(self):
        """向量库只找到部分 ID 时，只补全找到的。"""
        store_results = [
            QueryResult(id="c1", score=0.0, text="found text", metadata={}),
            # c2 不在返回结果中（向量库中不存在）
        ]
        hs, mock_store = self._make_hybrid_search(store_results=store_results)

        results = [
            RetrievalResult(chunk_id="c1", score=0.9, text="", metadata={}),
            RetrievalResult(chunk_id="c2", score=0.8, text="", metadata={}),
        ]

        filled = hs._fill_missing_text(results)

        assert filled[0].text == "found text"  # c1 被补全
        assert filled[1].text == ""  # c2 仍为空（向量库中不存在）

    def test_existing_metadata_not_overwritten(self):
        """如果结果已有 metadata，不会被覆盖。"""
        store_results = [
            QueryResult(id="c1", score=0.0, text="text", metadata={"new_key": "new_val"}),
        ]
        hs, mock_store = self._make_hybrid_search(store_results=store_results)

        results = [
            RetrievalResult(chunk_id="c1", score=0.9, text="", metadata={"existing": "data"}),
        ]

        filled = hs._fill_missing_text(results)

        assert filled[0].text == "text"  # text 被补全
        assert filled[0].metadata == {"existing": "data"}  # metadata 保持不变（已有值）


class TestHybridSearchIntegration:
    """验证 HybridSearch.search() 的完整流程。"""

    def test_search_calls_fill_missing_text(self):
        """验证 search() 在 fusion 之后调用了 text 补全。"""
        mock_settings = MagicMock()
        mock_settings.retrieval.rrf_k = 60
        mock_settings.retrieval.top_k = 5

        mock_processor = MagicMock()
        mock_processor.process.return_value = MagicMock(
            keywords=["python"], filters=None
        )

        # Dense 返回有 text 的结果
        dense_result = RetrievalResult(chunk_id="c1", score=0.9, text="dense text", metadata={})
        mock_dense = MagicMock()
        mock_dense.retrieve.return_value = [dense_result]

        # Sparse 返回无 text 的结果
        sparse_result = RetrievalResult(chunk_id="c2", score=0.8, text="", metadata={})
        mock_sparse = MagicMock()
        mock_sparse.retrieve.return_value = [sparse_result]

        # Fusion 返回合并结果
        mock_fusion = MagicMock()
        mock_fusion.fuse.return_value = [
            RetrievalResult(chunk_id="c1", score=0.03, text="dense text", metadata={}),
            RetrievalResult(chunk_id="c2", score=0.02, text="", metadata={}),
        ]

        # 向量库补全 c2 的 text
        mock_store = MagicMock()
        mock_store.get_by_ids.return_value = [
            QueryResult(id="c2", score=0.0, text="sparse filled", metadata={"k": "v"}),
        ]

        hs = HybridSearch(
            settings=mock_settings,
            query_processor=mock_processor,
            dense_retriever=mock_dense,
            sparse_retriever=mock_sparse,
            fusion=mock_fusion,
            vector_store=mock_store,
        )

        results = hs.search("python programming")

        assert len(results) == 2
        assert results[0].text == "dense text"
        assert results[1].text == "sparse filled"
        mock_store.get_by_ids.assert_called_once_with(["c2"])
