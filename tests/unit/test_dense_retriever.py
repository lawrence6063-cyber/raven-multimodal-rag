"""Tests for DenseRetriever."""

import pytest
from unittest.mock import MagicMock

from src.core.types import RetrievalResult
from src.core.query_engine.dense_retriever import DenseRetriever
from src.core.settings import Settings
from src.libs.vector_store.base_vector_store import QueryResult


class TestDenseRetriever:
    def test_retrieve_success(self):
        settings = Settings()
        mock_embedding = MagicMock()
        mock_embedding.embed.return_value = [[0.1, 0.2, 0.3]]

        mock_store = MagicMock()
        mock_store.query.return_value = [
            QueryResult(id="c1", score=0.95, text="chunk one", metadata={"source": "a.pdf"}),
            QueryResult(id="c2", score=0.88, text="chunk two", metadata={}),
        ]

        retriever = DenseRetriever(settings, embedding_client=mock_embedding, vector_store=mock_store)
        results = retriever.retrieve("test query", top_k=5)

        assert len(results) == 2
        assert results[0].chunk_id == "c1"
        assert results[0].score == 0.95
        assert results[0].text == "chunk one"
        mock_embedding.embed.assert_called_once_with(["test query"])

    def test_retrieve_empty_embedding(self):
        settings = Settings()
        mock_embedding = MagicMock()
        mock_embedding.embed.return_value = [[]]

        retriever = DenseRetriever(settings, embedding_client=mock_embedding, vector_store=MagicMock())
        results = retriever.retrieve("query")
        assert results == []

    def test_passes_filters_to_store(self):
        settings = Settings()
        mock_embedding = MagicMock()
        mock_embedding.embed.return_value = [[0.1]]
        mock_store = MagicMock()
        mock_store.query.return_value = []

        retriever = DenseRetriever(settings, embedding_client=mock_embedding, vector_store=mock_store)
        retriever.retrieve("q", filters={"collection": "test"})

        mock_store.query.assert_called_once()
        call_kwargs = mock_store.query.call_args
        assert call_kwargs[1]["filters"] == {"collection": "test"}
