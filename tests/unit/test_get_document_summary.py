"""Unit tests for GetDocumentSummaryTool (E5)."""

from __future__ import annotations

import pytest

from src.mcp_server.protocol_handler import INVALID_PARAMS, JsonRpcError
from src.mcp_server.tools.get_document_summary import GetDocumentSummaryTool


def test_returns_structured_summary_when_found():
    provider = lambda doc_id: {
        "title": "Intro to RAG",
        "summary": "A primer on retrieval augmented generation.",
        "tags": ["rag", "nlp"],
        "created_at": "2026-06-10",
    }
    result = GetDocumentSummaryTool(provider).run("docA")
    sc = result.structured_content
    assert sc["doc_id"] == "docA"
    assert sc["title"] == "Intro to RAG"
    assert sc["tags"] == ["rag", "nlp"]
    assert "RAG" in result.content[0].text


def test_tags_string_is_split():
    provider = lambda doc_id: {"title": "t", "summary": "s", "tags": "a, b ,c"}
    result = GetDocumentSummaryTool(provider).run("docB")
    assert result.structured_content["tags"] == ["a", "b", "c"]


def test_missing_document_raises_invalid_params():
    provider = lambda doc_id: None
    with pytest.raises(JsonRpcError) as exc:
        GetDocumentSummaryTool(provider).run("ghost")
    assert exc.value.code == INVALID_PARAMS
    assert "ghost" in exc.value.message


def test_untitled_fallback():
    provider = lambda doc_id: {"summary": "", "tags": []}
    result = GetDocumentSummaryTool(provider).run("docC")
    assert result.structured_content["title"] == "(untitled)"
