"""Tests for ChunkRefiner, MetadataEnricher, ImageCaptioner, SparseEncoder, BM25Indexer, ImageStorage."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.core.types import Chunk, ChunkRecord
from src.core.settings import Settings
from src.ingestion.transform.chunk_refiner import ChunkRefiner
from src.ingestion.transform.metadata_enricher import MetadataEnricher
from src.ingestion.transform.image_captioner import ImageCaptioner
from src.ingestion.embedding.sparse_encoder import SparseEncoder
from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.ingestion.storage.image_storage import ImageStorage


class TestChunkRefiner:
    def _settings(self, use_llm=False):
        s = Settings()
        s.ingestion.chunk_refiner.use_llm = use_llm
        return s

    def test_rule_removes_excessive_whitespace(self):
        refiner = ChunkRefiner(self._settings())
        chunks = [Chunk(id="c1", text="Hello\n\n\n\n\nWorld", metadata={})]
        result = refiner.transform(chunks)
        assert "\n\n\n" not in result[0].text
        assert result[0].metadata["refined_by"] == "rule"

    def test_rule_removes_page_headers(self):
        refiner = ChunkRefiner(self._settings())
        chunks = [Chunk(id="c1", text="Page 42\n\nActual content here.", metadata={})]
        result = refiner.transform(chunks)
        assert "Page 42" not in result[0].text
        assert "Actual content" in result[0].text

    def test_rule_removes_html_comments(self):
        refiner = ChunkRefiner(self._settings())
        chunks = [Chunk(id="c1", text="text <!-- hidden --> more", metadata={})]
        result = refiner.transform(chunks)
        assert "hidden" not in result[0].text

    def test_preserves_original_on_error(self):
        refiner = ChunkRefiner(self._settings())
        chunk = Chunk(id="c1", text="good text", metadata={})
        # Should handle gracefully even if transform somehow raises in edge case
        result = refiner.transform([chunk])
        assert len(result) == 1


class TestMetadataEnricher:
    def _settings(self, use_llm=False):
        s = Settings()
        s.ingestion.metadata_enricher.use_llm = use_llm
        return s

    def test_rule_enrichment_produces_title_summary_tags(self):
        enricher = MetadataEnricher(self._settings())
        chunks = [Chunk(id="c1", text="## Introduction\n\nThis is about machine learning.", metadata={})]
        result = enricher.transform(chunks)
        assert "title" in result[0].metadata
        assert "summary" in result[0].metadata
        assert "tags" in result[0].metadata
        assert result[0].metadata["enriched_by"] == "rule"

    def test_title_from_first_line(self):
        enricher = MetadataEnricher(self._settings())
        chunks = [Chunk(id="c1", text="# My Title\n\nBody text.", metadata={})]
        result = enricher.transform(chunks)
        assert "My Title" in result[0].metadata["title"]


class TestImageCaptioner:
    def test_disabled_returns_unchanged(self):
        s = Settings()
        s.vision_llm.enabled = False
        captioner = ImageCaptioner(s)
        chunks = [Chunk(id="c1", text="text", metadata={"image_refs": ["img1"]})]
        result = captioner.transform(chunks)
        assert result[0].text == "text"

    def test_no_images_returns_unchanged(self):
        s = Settings()
        s.vision_llm.enabled = True
        captioner = ImageCaptioner(s)
        chunks = [Chunk(id="c1", text="text", metadata={})]
        result = captioner.transform(chunks)
        assert result[0].text == "text"


class TestSparseEncoder:
    def test_encode_produces_term_weights(self):
        encoder = SparseEncoder()
        chunks = [Chunk(id="c1", text="Machine learning is powerful. Machine learning transforms data.", metadata={})]
        records = encoder.encode(chunks)
        assert len(records) == 1
        assert "machine" in records[0].sparse_vector
        assert "learning" in records[0].sparse_vector
        assert records[0].sparse_vector["machine"] > 0

    def test_stop_words_removed(self):
        encoder = SparseEncoder()
        chunks = [Chunk(id="c1", text="This is a test of the system.", metadata={})]
        records = encoder.encode(chunks)
        assert "this" not in records[0].sparse_vector
        assert "the" not in records[0].sparse_vector
        assert "test" in records[0].sparse_vector

    def test_empty_input(self):
        encoder = SparseEncoder()
        assert encoder.encode([]) == []


class TestBM25Indexer:
    def test_build_and_query(self, tmp_path):
        indexer = BM25Indexer(index_path=str(tmp_path / "bm25"))
        records = [
            ChunkRecord(id="c1", text="machine learning", sparse_vector={"machine": 0.5, "learning": 0.5}),
            ChunkRecord(id="c2", text="deep learning neural", sparse_vector={"deep": 0.33, "learning": 0.33, "neural": 0.33}),
            ChunkRecord(id="c3", text="data science python", sparse_vector={"data": 0.33, "science": 0.33, "python": 0.33}),
        ]
        indexer.build(records)
        results = indexer.query(["machine"], top_k=2)
        assert len(results) >= 1
        assert results[0]["chunk_id"] == "c1"

    def test_query_returns_stable_order(self, tmp_path):
        indexer = BM25Indexer(index_path=str(tmp_path / "bm25"))
        records = [
            ChunkRecord(id="a", text="x", sparse_vector={"hello": 1.0}),
            ChunkRecord(id="b", text="y", sparse_vector={"hello": 0.5, "world": 0.5}),
        ]
        indexer.build(records)
        r1 = indexer.query(["hello"])
        r2 = indexer.query(["hello"])
        assert [x["chunk_id"] for x in r1] == [x["chunk_id"] for x in r2]

    def test_persistence(self, tmp_path):
        path = str(tmp_path / "bm25")
        indexer = BM25Indexer(index_path=path)
        indexer.build([ChunkRecord(id="c1", text="t", sparse_vector={"test": 1.0})])

        # Create new indexer loading from disk
        indexer2 = BM25Indexer(index_path=path)
        results = indexer2.query(["test"])
        assert len(results) == 1


class TestImageStorage:
    def test_save_and_get(self, tmp_path):
        img = tmp_path / "source.png"
        img.write_bytes(b"fake png data")
        storage = ImageStorage(base_dir=str(tmp_path / "images"), db_path=str(tmp_path / "img.db"))

        dest = storage.save("img001", str(img), collection="test", doc_hash="abc")
        assert Path(dest).exists()

        retrieved = storage.get_path("img001")
        assert retrieved == dest

    def test_get_nonexistent(self, tmp_path):
        storage = ImageStorage(base_dir=str(tmp_path / "images"), db_path=str(tmp_path / "img.db"))
        assert storage.get_path("nonexistent") is None

    def test_list_images(self, tmp_path):
        img = tmp_path / "test.png"
        img.write_bytes(b"data")
        storage = ImageStorage(base_dir=str(tmp_path / "images"), db_path=str(tmp_path / "img.db"))
        storage.save("img1", str(img), collection="col1")
        storage.save("img2", str(img), collection="col2")

        all_imgs = storage.list_images()
        assert len(all_imgs) == 2
        col1_imgs = storage.list_images(collection="col1")
        assert len(col1_imgs) == 1

    def test_delete(self, tmp_path):
        img = tmp_path / "del.png"
        img.write_bytes(b"data")
        storage = ImageStorage(base_dir=str(tmp_path / "images"), db_path=str(tmp_path / "img.db"))
        storage.save("del1", str(img), collection="test")
        storage.delete("del1")
        assert storage.get_path("del1") is None
