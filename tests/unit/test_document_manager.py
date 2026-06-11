"""Unit tests for DocumentManager (G2)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.ingestion.document_manager import (
    CollectionStats,
    DeleteResult,
    DocumentInfo,
    DocumentManager,
)


def _build_manager():
    """Build a DocumentManager with all backends mocked."""
    chroma = MagicMock()
    bm25 = MagicMock()
    images = MagicMock()
    integrity = MagicMock()

    # Default list_processed returns one document
    integrity.list_processed.return_value = [
        {
            "file_hash": "abc123",
            "file_path": "data/documents/kb1/guide.pdf",
            "processed_at": "2026-06-10T10:00:00",
            "chunk_count": 5,
        }
    ]

    # Default list_images
    images.list_images.return_value = [
        {"image_id": "img1", "file_path": "/tmp/img.png", "collection": "kb1", "doc_hash": "abc123", "page_num": 1}
    ]

    return DocumentManager(chroma, bm25, images, integrity), chroma, bm25, images, integrity


class TestListDocuments:
    def test_returns_all_documents(self):
        mgr, *_ = _build_manager()
        docs = mgr.list_documents()
        assert len(docs) == 1
        assert docs[0].source_path == "data/documents/kb1/guide.pdf"
        assert docs[0].collection == "kb1"
        assert docs[0].chunk_count == 5
        assert docs[0].image_count == 1

    def test_filter_by_collection(self):
        mgr, _, _, images, integrity = _build_manager()
        # Add a second doc in different collection
        integrity.list_processed.return_value = [
            {"file_hash": "abc", "file_path": "data/documents/kb1/a.pdf", "processed_at": "", "chunk_count": 3},
            {"file_hash": "def", "file_path": "data/documents/kb2/b.pdf", "processed_at": "", "chunk_count": 2},
        ]
        images.list_images.return_value = []

        docs = mgr.list_documents(collection="kb2")
        assert len(docs) == 1
        assert docs[0].collection == "kb2"

    def test_empty_history(self):
        mgr, _, _, images, integrity = _build_manager()
        integrity.list_processed.return_value = []
        images.list_images.return_value = []
        assert mgr.list_documents() == []


class TestDeleteDocument:
    def test_successful_deletion(self):
        mgr, chroma, bm25, images, integrity = _build_manager()
        # Chroma returns chunks for this source_path
        chroma.get_by_metadata.return_value = [
            SimpleNamespace(id="c1", text="t", metadata={}),
            SimpleNamespace(id="c2", text="t", metadata={}),
        ]
        images.list_images.return_value = [
            {"image_id": "img1", "doc_hash": "abc123"},
        ]

        result = mgr.delete_document("data/documents/kb1/guide.pdf")

        assert result.success is True
        assert result.chunks_deleted == 2
        assert result.images_deleted == 1
        chroma.delete.assert_called_once_with(["c1", "c2"])
        bm25.remove_document.assert_called_once_with(["c1", "c2"])
        images.delete.assert_called_once_with("img1")
        integrity.remove_record.assert_called_once_with("abc123")

    def test_not_found(self):
        mgr, _, _, _, integrity = _build_manager()
        integrity.list_processed.return_value = []

        result = mgr.delete_document("nonexistent.pdf")

        assert result.success is False
        assert "not found" in result.error.lower()


class TestGetDocumentDetail:
    def test_returns_detail(self):
        mgr, chroma, _, images, _ = _build_manager()
        chroma.get_by_metadata.return_value = [
            SimpleNamespace(
                id="c1", text="chunk text here",
                metadata={"collection": "kb1", "source_path": "guide.pdf", "image_refs": ["img1"]},
            )
        ]
        images.get_path.return_value = "/tmp/img1.png"

        detail = mgr.get_document_detail("doc_001")

        assert detail is not None
        assert detail.doc_id == "doc_001"
        assert detail.chunk_count == 1
        assert detail.image_count == 1

    def test_not_found_returns_none(self):
        mgr, chroma, _, _, _ = _build_manager()
        chroma.get_by_metadata.return_value = []

        assert mgr.get_document_detail("missing") is None


class TestCollectionStats:
    def test_stats(self):
        mgr, *_ = _build_manager()
        stats = mgr.get_collection_stats()
        assert stats.total_documents == 1
        assert stats.total_chunks == 5
        assert stats.total_images == 1
