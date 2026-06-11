"""Tests for Embedding factory and base interface."""

import pytest

from src.libs.embedding.base_embedding import BaseEmbedding, EmbeddingError
from src.libs.embedding.embedding_factory import EmbeddingFactory, register_embedding, _EMBEDDING_REGISTRY
from src.core.settings import EmbeddingSettings


class FakeEmbedding(BaseEmbedding):
    """Fake embedding for testing."""

    def __init__(self, settings=None):
        self._dims = 128

    def embed(self, texts):
        return [[0.1] * self._dims for _ in texts]

    @property
    def provider_name(self):
        return "fake"

    @property
    def dimensions(self):
        return self._dims


class TestEmbeddingFactory:
    """Test EmbeddingFactory routing logic."""

    def setup_method(self):
        self._original = _EMBEDDING_REGISTRY.copy()
        _EMBEDDING_REGISTRY["fake"] = FakeEmbedding

    def teardown_method(self):
        _EMBEDDING_REGISTRY.clear()
        _EMBEDDING_REGISTRY.update(self._original)

    def test_create_known_provider(self):
        settings = EmbeddingSettings(provider="fake", model="test")
        emb = EmbeddingFactory.create(settings)
        assert isinstance(emb, FakeEmbedding)

    def test_create_unknown_provider_raises(self):
        settings = EmbeddingSettings(provider="nonexistent", model="test")
        with pytest.raises(EmbeddingError, match="Unknown embedding provider"):
            EmbeddingFactory.create(settings)

    def test_embed_returns_correct_shape(self):
        emb = FakeEmbedding()
        vectors = emb.embed(["hello", "world"])
        assert len(vectors) == 2
        assert len(vectors[0]) == 128

    def test_dimensions_property(self):
        emb = FakeEmbedding()
        assert emb.dimensions == 128

    def test_empty_input(self):
        emb = FakeEmbedding()
        vectors = emb.embed([])
        assert vectors == []
