"""Tests for VectorStore contract and factory."""

import json

import pytest

from src.libs.vector_store.base_vector_store import (
    BaseVectorStore,
    VectorRecord,
    QueryResult,
    VectorStoreError,
)
from src.libs.vector_store.vector_store_factory import (
    VectorStoreFactory,
    register_vector_store,
    _VECTOR_STORE_REGISTRY,
)
from src.core.settings import VectorStoreSettings


class FakeVectorStore(BaseVectorStore):
    """In-memory vector store for testing."""

    def __init__(self, settings=None):
        self._store: dict[str, VectorRecord] = {}

    def upsert(self, records):
        for r in records:
            self._store[r.id] = r

    def query(self, vector, top_k=10, filters=None):
        # Return all stored records sorted by ID (deterministic for tests)
        results = []
        for rec in list(self._store.values())[:top_k]:
            results.append(QueryResult(id=rec.id, score=0.9, text=rec.text, metadata=rec.metadata))
        return results

    def delete(self, ids):
        for id_ in ids:
            self._store.pop(id_, None)

    def delete_by_metadata(self, where):
        # Mirror ChromaStore semantics: empty filter deletes nothing.
        if not where:
            return 0
        matched = [
            rec_id
            for rec_id, rec in self._store.items()
            if all(rec.metadata.get(k) == v for k, v in where.items())
        ]
        for rec_id in matched:
            self._store.pop(rec_id, None)
        return len(matched)

    def get_collection_stats(self):
        return {
            "collection_name": "fake",
            "persist_directory": ":memory:",
            "total_chunks": len(self._store),
        }

    def get_by_ids(self, ids):
        results = []
        for id_ in ids:
            if id_ in self._store:
                rec = self._store[id_]
                results.append(QueryResult(id=rec.id, score=0.0, text=rec.text, metadata=rec.metadata))
        return results

    @property
    def provider_name(self):
        return "fake"


class TestVectorStoreFactory:
    """Test VectorStoreFactory routing."""

    def setup_method(self):
        self._original = _VECTOR_STORE_REGISTRY.copy()
        _VECTOR_STORE_REGISTRY["fake"] = FakeVectorStore

    def teardown_method(self):
        _VECTOR_STORE_REGISTRY.clear()
        _VECTOR_STORE_REGISTRY.update(self._original)

    def test_create_known_provider(self):
        settings = VectorStoreSettings(provider="fake")
        store = VectorStoreFactory.create(settings)
        assert isinstance(store, FakeVectorStore)

    def test_create_unknown_provider_raises(self):
        settings = VectorStoreSettings(provider="nonexistent")
        with pytest.raises(VectorStoreError, match="Unknown vector store provider"):
            VectorStoreFactory.create(settings)


class TestVectorStoreContract:
    """Test VectorStore input/output shape contract."""

    def test_upsert_and_query_roundtrip(self):
        store = FakeVectorStore()
        records = [
            VectorRecord(id="r1", vector=[0.1, 0.2], text="hello", metadata={"k": "v"}),
            VectorRecord(id="r2", vector=[0.3, 0.4], text="world"),
        ]
        store.upsert(records)
        results = store.query(vector=[0.1, 0.2], top_k=5)
        assert len(results) == 2
        assert all(isinstance(r, QueryResult) for r in results)
        assert results[0].text in ("hello", "world")

    def test_delete_removes_records(self):
        store = FakeVectorStore()
        store.upsert([VectorRecord(id="d1", vector=[1.0], text="delete me")])
        store.delete(["d1"])
        results = store.query(vector=[1.0])
        assert len(results) == 0

    def test_query_respects_top_k(self):
        store = FakeVectorStore()
        records = [VectorRecord(id=f"r{i}", vector=[float(i)], text=f"doc{i}") for i in range(10)]
        store.upsert(records)
        results = store.query(vector=[0.0], top_k=3)
        assert len(results) == 3

    def test_vector_record_structure(self):
        rec = VectorRecord(id="test", vector=[1.0, 2.0], text="content", metadata={"page": 1})
        assert rec.id == "test"
        assert rec.vector == [1.0, 2.0]
        assert rec.metadata["page"] == 1

    def test_get_by_ids_returns_existing_records(self):
        store = FakeVectorStore()
        store.upsert([
            VectorRecord(id="g1", vector=[1.0], text="text1", metadata={"k": "v1"}),
            VectorRecord(id="g2", vector=[2.0], text="text2", metadata={"k": "v2"}),
        ])
        results = store.get_by_ids(["g1", "g2"])
        assert len(results) == 2
        assert all(isinstance(r, QueryResult) for r in results)
        assert results[0].id == "g1"
        assert results[0].text == "text1"
        assert results[0].score == 0.0
        assert results[1].metadata == {"k": "v2"}

    def test_get_by_ids_skips_missing(self):
        store = FakeVectorStore()
        store.upsert([VectorRecord(id="exists", vector=[1.0], text="hello")])
        results = store.get_by_ids(["exists", "not_exists"])
        assert len(results) == 1
        assert results[0].id == "exists"

    def test_get_by_ids_empty_list(self):
        store = FakeVectorStore()
        results = store.get_by_ids([])
        assert results == []


class TestDeleteByMetadataContract:
    """Boundary contract for delete_by_metadata (ChromaStore extension)."""

    def _seed(self) -> FakeVectorStore:
        store = FakeVectorStore()
        store.upsert([
            VectorRecord(id="a1", vector=[0.1], text="alpha", metadata={"doc_id": "D1", "page": 1}),
            VectorRecord(id="a2", vector=[0.2], text="beta", metadata={"doc_id": "D1", "page": 2}),
            VectorRecord(id="b1", vector=[0.3], text="gamma", metadata={"doc_id": "D2", "page": 1}),
        ])
        return store

    def test_empty_filter_deletes_nothing(self):
        store = self._seed()
        deleted = store.delete_by_metadata({})
        assert deleted == 0
        assert store.get_collection_stats()["total_chunks"] == 3

    def test_matching_filter_deletes_all_matches(self):
        store = self._seed()
        deleted = store.delete_by_metadata({"doc_id": "D1"})
        assert deleted == 2
        remaining = store.get_by_ids(["a1", "a2", "b1"])
        assert [r.id for r in remaining] == ["b1"]

    def test_no_match_deletes_nothing(self):
        store = self._seed()
        deleted = store.delete_by_metadata({"doc_id": "missing"})
        assert deleted == 0
        assert store.get_collection_stats()["total_chunks"] == 3

    def test_multi_key_filter_requires_all(self):
        store = self._seed()
        deleted = store.delete_by_metadata({"doc_id": "D1", "page": 2})
        assert deleted == 1
        assert store.get_by_ids(["a2"]) == []


class TestCollectionStatsContract:
    """Contract for get_collection_stats output shape."""

    def test_stats_shape_and_count(self):
        store = FakeVectorStore()
        store.upsert([VectorRecord(id="s1", vector=[1.0], text="x")])
        stats = store.get_collection_stats()
        assert set(stats) >= {"collection_name", "persist_directory", "total_chunks"}
        assert stats["total_chunks"] == 1

    def test_stats_empty_store(self):
        stats = FakeVectorStore().get_collection_stats()
        assert stats["total_chunks"] == 0


class TestChromaMetadataSanitize:
    """ChromaStore._sanitize_metadata coerces values into Chroma-safe forms."""

    def _sanitize(self, metadata):
        from src.libs.vector_store.chroma_store import ChromaStore
        return ChromaStore._sanitize_metadata(metadata)

    def test_none_and_empty_metadata(self):
        assert self._sanitize(None) == {}
        assert self._sanitize({}) == {}

    def test_drops_empty_list_and_none_values(self):
        out = self._sanitize({"tags": [], "note": None, "page": 3})
        assert "tags" not in out          # empty list dropped (the original bug)
        assert "note" not in out          # None dropped
        assert out["page"] == 3

    def test_keeps_scalars(self):
        out = self._sanitize({"s": "a", "i": 1, "f": 1.5, "b": True})
        assert out == {"s": "a", "i": 1, "f": 1.5, "b": True}

    def test_keeps_nonempty_scalar_list(self):
        # image_refs must stay a list for the multimodal assembler.
        out = self._sanitize({"image_refs": ["fig1", "fig2"], "tags": ["A", "B"]})
        assert out["image_refs"] == ["fig1", "fig2"]
        assert out["tags"] == ["A", "B"]

    def test_json_encodes_list_of_dicts(self):
        out = self._sanitize({"images": [{"id": "x", "path": "/p"}]})
        assert isinstance(out["images"], str)
        assert json.loads(out["images"]) == [{"id": "x", "path": "/p"}]

    def test_json_encodes_dict(self):
        out = self._sanitize({"extra": {"k": "v"}})
        assert json.loads(out["extra"]) == {"k": "v"}
