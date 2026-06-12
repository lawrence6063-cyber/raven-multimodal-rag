"""Tests for DocumentChunker."""

import pytest
from unittest.mock import patch, MagicMock

from src.core.types import Document, Chunk
from src.ingestion.chunking.document_chunker import DocumentChunker
from src.core.settings import Settings, SplitterSettings


class TestDocumentChunker:
    """Test DocumentChunker behavior."""

    def _make_settings(self, chunk_size=50, chunk_overlap=10):
        s = Settings()
        s.splitter = SplitterSettings(provider="fake", chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return s

    @patch("src.ingestion.chunking.document_chunker.SplitterFactory")
    def test_split_produces_chunks(self, mock_factory):
        mock_splitter = MagicMock()
        mock_splitter.split_text.return_value = ["chunk one", "chunk two", "chunk three"]
        mock_factory.create.return_value = mock_splitter

        settings = self._make_settings()
        chunker = DocumentChunker(settings)
        doc = Document(id="doc1", text="full text", metadata={"source_path": "test.pdf", "doc_type": "pdf"})
        chunks = chunker.split_document(doc)

        assert len(chunks) == 3
        assert all(isinstance(c, Chunk) for c in chunks)

    @patch("src.ingestion.chunking.document_chunker.SplitterFactory")
    def test_chunk_ids_are_unique(self, mock_factory):
        mock_splitter = MagicMock()
        mock_splitter.split_text.return_value = ["a", "b", "c"]
        mock_factory.create.return_value = mock_splitter

        chunker = DocumentChunker(self._make_settings())
        doc = Document(id="doc1", text="text", metadata={})
        chunks = chunker.split_document(doc)

        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids))

    @patch("src.ingestion.chunking.document_chunker.SplitterFactory")
    def test_chunk_ids_are_deterministic(self, mock_factory):
        mock_splitter = MagicMock()
        mock_splitter.split_text.return_value = ["same text"]
        mock_factory.create.return_value = mock_splitter

        chunker = DocumentChunker(self._make_settings())
        doc = Document(id="doc1", text="t", metadata={})
        c1 = chunker.split_document(doc)
        c2 = chunker.split_document(doc)
        assert c1[0].id == c2[0].id

    @patch("src.ingestion.chunking.document_chunker.SplitterFactory")
    def test_metadata_inherited(self, mock_factory):
        mock_splitter = MagicMock()
        mock_splitter.split_text.return_value = ["chunk"]
        mock_factory.create.return_value = mock_splitter

        chunker = DocumentChunker(self._make_settings())
        doc = Document(id="doc1", text="t", metadata={"source_path": "a.pdf", "doc_type": "pdf", "title": "Test"})
        chunks = chunker.split_document(doc)

        assert chunks[0].metadata["source_path"] == "a.pdf"
        assert chunks[0].metadata["doc_type"] == "pdf"
        assert chunks[0].metadata["title"] == "Test"
        assert chunks[0].metadata["chunk_index"] == 0

    @patch("src.ingestion.chunking.document_chunker.SplitterFactory")
    def test_source_ref_links_to_document(self, mock_factory):
        mock_splitter = MagicMock()
        mock_splitter.split_text.return_value = ["chunk"]
        mock_factory.create.return_value = mock_splitter

        chunker = DocumentChunker(self._make_settings())
        doc = Document(id="my_doc_id", text="t", metadata={})
        chunks = chunker.split_document(doc)
        assert chunks[0].source_ref == "my_doc_id"

    @patch("src.ingestion.chunking.document_chunker.SplitterFactory")
    def test_image_refs_distributed(self, mock_factory):
        mock_splitter = MagicMock()
        mock_splitter.split_text.return_value = [
            "text before [IMAGE: img1] after",
            "no images here",
            "another [IMAGE: img2] section",
        ]
        mock_factory.create.return_value = mock_splitter

        images = [
            {"id": "img1", "path": "data/images/img1.png", "page": 1, "text_offset": 10, "text_length": 15},
            {"id": "img2", "path": "data/images/img2.png", "page": 2, "text_offset": 100, "text_length": 15},
        ]
        chunker = DocumentChunker(self._make_settings())
        doc = Document(id="doc1", text="full", metadata={"images": images})
        chunks = chunker.split_document(doc)

        # First chunk has img1
        assert chunks[0].metadata.get("image_refs") == ["img1"]
        assert len(chunks[0].metadata["images"]) == 1
        # Second chunk has no images
        assert "image_refs" not in chunks[1].metadata
        assert "images" not in chunks[1].metadata
        # Third chunk has img2
        assert chunks[2].metadata.get("image_refs") == ["img2"]

    @patch("src.ingestion.chunking.document_chunker.SplitterFactory")
    def test_empty_text_returns_empty(self, mock_factory):
        mock_splitter = MagicMock()
        mock_splitter.split_text.return_value = []
        mock_factory.create.return_value = mock_splitter

        chunker = DocumentChunker(self._make_settings())
        doc = Document(id="doc1", text="", metadata={})
        assert chunker.split_document(doc) == []

    @patch("src.ingestion.chunking.document_chunker.SplitterFactory")
    def test_block_type_classification(self, mock_factory):
        mock_splitter = MagicMock()
        mock_splitter.split_text.return_value = [
            "| Model | BLEU |\n| --- | --- |\n| A | 26.3 |",  # table
            "plain paragraph text",                              # text
            "see [IMAGE: img1] here",                            # image
            "# Section Title",                                   # heading
        ]
        mock_factory.create.return_value = mock_splitter

        chunker = DocumentChunker(self._make_settings())
        doc = Document(id="doc1", text="full", metadata={})
        chunks = chunker.split_document(doc)

        assert chunks[0].metadata["block_type"] == "table"
        assert chunks[1].metadata["block_type"] == "text"
        assert chunks[2].metadata["block_type"] == "image"
        assert chunks[3].metadata["block_type"] == "heading"
