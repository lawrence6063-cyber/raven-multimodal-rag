"""Tests for QueryProcessor."""

import pytest
from src.core.query_engine.query_processor import QueryProcessor, ProcessedQuery


class TestQueryProcessor:
    def test_extract_keywords_english(self):
        proc = QueryProcessor()
        result = proc.process("How to configure Azure OpenAI?")
        assert "configure" in result.keywords
        assert "azure" in result.keywords
        assert "openai" in result.keywords
        assert "how" not in result.keywords
        assert "to" not in result.keywords

    def test_extract_keywords_chinese(self):
        proc = QueryProcessor()
        result = proc.process("如何 配置 向量数据库")
        # With space-separated Chinese, each word is a token
        assert "配置" in result.keywords
        assert "向量数据库" in result.keywords

    def test_keywords_deduplicated(self):
        proc = QueryProcessor()
        result = proc.process("machine learning machine learning")
        assert result.keywords.count("machine") == 1

    def test_inline_filter_parsing(self):
        proc = QueryProcessor()
        result = proc.process("how to deploy collection:finance")
        assert result.filters.get("collection") == "finance"
        assert "deploy" in result.keywords

    def test_external_filters_merged(self):
        proc = QueryProcessor()
        result = proc.process("test query", filters={"collection": "docs"})
        assert result.filters["collection"] == "docs"

    def test_empty_query(self):
        proc = QueryProcessor()
        result = proc.process("")
        assert result.keywords == []
        assert result.original == ""

    def test_preserves_original(self):
        proc = QueryProcessor()
        result = proc.process("My original question?")
        assert result.original == "My original question?"
