"""Tests for core data types (Document/Chunk/ChunkRecord/RetrievalResult)."""

import pytest
import json

from src.core.types import Document, Chunk, ChunkRecord, RetrievalResult


class TestDocument:
    def test_create_document(self):
        doc = Document(id="doc1", text="Hello world", metadata={"source_path": "test.pdf"})
        assert doc.id == "doc1"
        assert doc.text == "Hello world"
        assert doc.metadata["source_path"] == "test.pdf"

    def test_to_dict(self):
        doc = Document(id="d1", text="content", metadata={"k": "v"})
        d = doc.to_dict()
        assert d == {"id": "d1", "text": "content", "metadata": {"k": "v"}}

    def test_from_dict(self):
        data = {"id": "d2", "text": "text2", "metadata": {"page": 1}}
        doc = Document.from_dict(data)
        assert doc.id == "d2"
        assert doc.metadata["page"] == 1

    def test_serializable_json(self):
        doc = Document(id="d1", text="t", metadata={"a": [1, 2]})
        s = json.dumps(doc.to_dict())
        assert '"id": "d1"' in s


class TestChunk:
    def test_create_chunk(self):
        chunk = Chunk(id="c1", text="chunk text", metadata={"chunk_index": 0}, source_ref="doc1")
        assert chunk.id == "c1"
        assert chunk.source_ref == "doc1"

    def test_to_dict_roundtrip(self):
        chunk = Chunk(id="c1", text="t", metadata={"x": 1}, source_ref="d1")
        d = chunk.to_dict()
        c2 = Chunk.from_dict(d)
        assert c2.id == chunk.id
        assert c2.source_ref == "d1"

    def test_metadata_images_field(self):
        images = [{"id": "img1", "path": "data/images/img1.png", "page": 1, "text_offset": 50, "text_length": 15}]
        chunk = Chunk(id="c1", text="text [IMAGE: img1]", metadata={"images": images})
        assert chunk.metadata["images"][0]["id"] == "img1"


class TestChunkRecord:
    def test_create_with_vectors(self):
        rec = ChunkRecord(id="r1", text="text", dense_vector=[0.1, 0.2], sparse_vector={"hello": 1.5})
        assert rec.dense_vector == [0.1, 0.2]
        assert rec.sparse_vector["hello"] == 1.5

    def test_to_dict(self):
        rec = ChunkRecord(id="r1", text="t", metadata={}, dense_vector=[1.0], sparse_vector={"a": 2.0})
        d = rec.to_dict()
        assert d["dense_vector"] == [1.0]
        assert d["sparse_vector"] == {"a": 2.0}


class TestRetrievalResult:
    def test_create(self):
        r = RetrievalResult(chunk_id="c1", score=0.95, text="result text", metadata={"source": "doc.pdf"})
        assert r.chunk_id == "c1"
        assert r.score == 0.95
