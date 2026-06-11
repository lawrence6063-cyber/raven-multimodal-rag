"""Tests for Splitter factory and base interface."""

import pytest

from src.libs.splitter.base_splitter import BaseSplitter, SplitterError
from src.libs.splitter.splitter_factory import SplitterFactory, register_splitter, _SPLITTER_REGISTRY
from src.core.settings import SplitterSettings


class FakeSplitter(BaseSplitter):
    """Fake splitter for testing."""

    def __init__(self, settings=None):
        self.chunk_size = getattr(settings, "chunk_size", 100) if settings else 100

    def split_text(self, text):
        # Simple split by chunk_size
        chunks = []
        for i in range(0, len(text), self.chunk_size):
            chunks.append(text[i:i + self.chunk_size])
        return chunks if chunks else [text] if text else []

    @property
    def provider_name(self):
        return "fake"


class TestSplitterFactory:
    """Test SplitterFactory routing logic."""

    def setup_method(self):
        self._original = _SPLITTER_REGISTRY.copy()
        _SPLITTER_REGISTRY["fake"] = FakeSplitter

    def teardown_method(self):
        _SPLITTER_REGISTRY.clear()
        _SPLITTER_REGISTRY.update(self._original)

    def test_create_known_provider(self):
        settings = SplitterSettings(provider="fake", chunk_size=50)
        splitter = SplitterFactory.create(settings)
        assert isinstance(splitter, FakeSplitter)
        assert splitter.chunk_size == 50

    def test_create_unknown_provider_raises(self):
        settings = SplitterSettings(provider="nonexistent")
        with pytest.raises(SplitterError, match="Unknown splitter provider"):
            SplitterFactory.create(settings)

    def test_split_text_produces_chunks(self):
        splitter = FakeSplitter(SplitterSettings(provider="fake", chunk_size=10))
        chunks = splitter.split_text("Hello world, this is a test.")
        assert len(chunks) > 1
        assert "".join(chunks) == "Hello world, this is a test."

    def test_split_empty_text(self):
        splitter = FakeSplitter()
        chunks = splitter.split_text("")
        assert chunks == []
