"""Unit tests for ListCollectionsTool (E4)."""

from __future__ import annotations

import pytest

from src.mcp_server.tools.list_collections import ListCollectionsTool


@pytest.fixture
def docs_root(tmp_path):
    """A documents root with two collections and a stray file."""
    (tmp_path / "alpha").mkdir()
    (tmp_path / "alpha" / "a1.pdf").write_text("x")
    (tmp_path / "alpha" / "a2.pdf").write_text("y")
    (tmp_path / "beta").mkdir()
    (tmp_path / "beta" / "b1.md").write_text("z")
    (tmp_path / "loose.txt").write_text("ignored")  # not a collection
    return tmp_path


def test_lists_collection_names(docs_root):
    result = ListCollectionsTool(str(docs_root)).run()
    names = {c["name"] for c in result.structured_content["collections"]}
    assert names == {"alpha", "beta"}


def test_reports_document_counts(docs_root):
    result = ListCollectionsTool(str(docs_root)).run()
    by_name = {c["name"]: c["document_count"] for c in result.structured_content["collections"]}
    assert by_name == {"alpha": 2, "beta": 1}


def test_markdown_content_first(docs_root):
    result = ListCollectionsTool(str(docs_root)).run()
    assert result.content[0].type == "text"
    assert "alpha" in result.content[0].text


def test_missing_dir_returns_empty(tmp_path):
    result = ListCollectionsTool(str(tmp_path / "nope")).run()
    assert result.structured_content["collections"] == []
    assert result.is_error is False
    assert "No collections" in result.content[0].text
