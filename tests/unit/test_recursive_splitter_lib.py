"""Tests for RecursiveSplitter (libs layer)."""

import pytest

from src.libs.splitter.splitter_factory import SplitterFactory, _SPLITTER_REGISTRY
from src.core.settings import SplitterSettings

# Import to trigger registration
from src.libs.splitter.recursive_splitter import RecursiveSplitter


class TestRecursiveSplitter:
    """Test RecursiveSplitter implementation."""

    def test_factory_creates_recursive(self):
        settings = SplitterSettings(provider="recursive", chunk_size=200, chunk_overlap=50)
        splitter = SplitterFactory.create(settings)
        assert isinstance(splitter, RecursiveSplitter)
        assert splitter.provider_name == "recursive"

    def test_split_short_text_single_chunk(self):
        settings = SplitterSettings(provider="recursive", chunk_size=1000, chunk_overlap=100)
        splitter = RecursiveSplitter(settings)
        text = "This is a short paragraph that fits in one chunk."
        chunks = splitter.split_text(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_split_long_text_multiple_chunks(self):
        settings = SplitterSettings(provider="recursive", chunk_size=50, chunk_overlap=10)
        splitter = RecursiveSplitter(settings)
        text = "This is sentence one. " * 20  # ~440 chars
        chunks = splitter.split_text(text)
        assert len(chunks) > 1
        # All chunks should be non-empty
        assert all(c.strip() for c in chunks)

    def test_split_markdown_respects_headers(self):
        settings = SplitterSettings(provider="recursive", chunk_size=100, chunk_overlap=20)
        splitter = RecursiveSplitter(settings)
        text = """## Introduction

This is the introduction section with some content.

## Methods

This is the methods section with different content.

## Results

Here are the results of the experiment.
"""
        chunks = splitter.split_text(text)
        assert len(chunks) >= 2

    def test_split_empty_text(self):
        settings = SplitterSettings(provider="recursive", chunk_size=100, chunk_overlap=20)
        splitter = RecursiveSplitter(settings)
        assert splitter.split_text("") == []
        assert splitter.split_text("   ") == []

    def test_registered_in_factory(self):
        assert "recursive" in _SPLITTER_REGISTRY
