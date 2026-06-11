"""Unit tests for ResponseBuilder and CitationGenerator (E3)."""

from __future__ import annotations

import pytest

from src.core.response.citation_generator import CitationGenerator
from src.core.response.response_builder import ResponseBuilder
from src.core.types import RetrievalResult


@pytest.fixture
def sample_results() -> list[RetrievalResult]:
    """Two retrieval results with differing metadata shapes."""
    return [
        RetrievalResult(
            chunk_id="docA_0001_abcd",
            score=0.9123,
            text="The quick brown fox\njumps over the lazy dog.",
            metadata={"source_path": "docs/a.pdf", "page": 5, "doc_id": "docA"},
        ),
        RetrievalResult(
            chunk_id="docB_0002_efgh",
            score=0.4211,
            text="Second relevant passage.",
            metadata={"file_name": "b.md", "doc_id": "docB"},
        ),
    ]


class TestCitationGenerator:
    def test_generates_one_indexed_citation_per_result(self, sample_results):
        citations = CitationGenerator().generate(sample_results)
        assert [c.id for c in citations] == [1, 2]
        assert citations[0].chunk_id == "docA_0001_abcd"

    def test_resolves_source_and_page(self, sample_results):
        citations = CitationGenerator().generate(sample_results)
        assert citations[0].source == "docs/a.pdf"
        assert citations[0].page == 5
        # No page key -> None; falls back to file_name for source
        assert citations[1].source == "b.md"
        assert citations[1].page is None

    def test_unknown_source_when_missing(self):
        citations = CitationGenerator().generate(
            [RetrievalResult(chunk_id="x", score=0.1, text="t", metadata={})]
        )
        assert citations[0].source == "unknown"

    def test_text_preview_collapses_whitespace(self, sample_results):
        citations = CitationGenerator().generate(sample_results)
        assert "\n" not in citations[0].text
        assert citations[0].text.startswith("The quick brown fox")

    def test_empty_results_yield_empty_list(self):
        assert CitationGenerator().generate([]) == []


class TestResponseBuilder:
    def test_build_has_markdown_text_first(self, sample_results):
        result = ResponseBuilder().build(sample_results, "what is fox?")
        assert result.is_error is False
        assert result.content[0].type == "text"
        assert "[1]" in result.content[0].text
        assert "[2]" in result.content[0].text

    def test_build_structured_citations(self, sample_results):
        result = ResponseBuilder().build(sample_results, "q")
        citations = result.structured_content["citations"]
        assert len(citations) == 2
        first = citations[0]
        assert set(first.keys()) >= {"id", "source", "page", "chunk_id", "score"}
        assert first["chunk_id"] == "docA_0001_abcd"

    def test_empty_results_return_friendly_hint(self):
        result = ResponseBuilder().build([], "nothing")
        assert result.is_error is False
        assert "No relevant results" in result.content[0].text
        assert result.structured_content["citations"] == []

    def test_to_dict_shape(self, sample_results):
        payload = ResponseBuilder().build(sample_results, "q").to_dict()
        assert payload["content"][0]["type"] == "text"
        assert "structuredContent" in payload
        assert payload["isError"] is False
