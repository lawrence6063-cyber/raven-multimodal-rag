"""Tests for OpenAI/Azure Embedding providers (mock HTTP)."""

import pytest
from unittest.mock import patch, MagicMock

from src.libs.embedding.base_embedding import EmbeddingError
from src.libs.embedding.embedding_factory import EmbeddingFactory, _EMBEDDING_REGISTRY
from src.core.settings import EmbeddingSettings

# Import to trigger registration
from src.libs.embedding.openai_embedding import OpenAIEmbedding
from src.libs.embedding.azure_embedding import AzureEmbedding
from src.libs.embedding.ollama_embedding import OllamaEmbedding
from src.libs.embedding.qwen_embedding import QwenEmbedding


class TestOpenAIEmbedding:
    """Test OpenAI Embedding with mocked API."""

    def test_factory_creates_openai(self):
        settings = EmbeddingSettings(provider="openai", model="text-embedding-ada-002", api_key="sk-test")
        emb = EmbeddingFactory.create(settings)
        assert isinstance(emb, OpenAIEmbedding)

    @patch("openai.OpenAI")
    def test_embed_success(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_item1 = MagicMock(embedding=[0.1, 0.2, 0.3])
        mock_item2 = MagicMock(embedding=[0.4, 0.5, 0.6])
        mock_client.embeddings.create.return_value = MagicMock(data=[mock_item1, mock_item2])

        settings = EmbeddingSettings(provider="openai", model="ada", api_key="sk-test", dimensions=3)
        emb = OpenAIEmbedding(settings)
        vectors = emb.embed(["hello", "world"])

        assert len(vectors) == 2
        assert vectors[0] == [0.1, 0.2, 0.3]

    def test_embed_empty_input(self):
        settings = EmbeddingSettings(provider="openai", model="ada", api_key="sk-test")
        emb = OpenAIEmbedding(settings)
        assert emb.embed([]) == []

    def test_dimensions_property(self):
        settings = EmbeddingSettings(provider="openai", model="ada", api_key="sk-test", dimensions=1536)
        emb = OpenAIEmbedding(settings)
        assert emb.dimensions == 1536


class TestAzureEmbedding:
    """Test Azure Embedding with mocked API."""

    def test_factory_creates_azure(self):
        settings = EmbeddingSettings(provider="azure", model="ada", api_key="key", azure_endpoint="https://test.openai.azure.com")
        emb = EmbeddingFactory.create(settings)
        assert isinstance(emb, AzureEmbedding)
        assert emb.provider_name == "azure"


class TestOllamaEmbedding:
    """Test Ollama Embedding."""

    def test_factory_creates_ollama(self):
        settings = EmbeddingSettings(provider="ollama", model="nomic-embed-text", dimensions=768)
        emb = EmbeddingFactory.create(settings)
        assert isinstance(emb, OllamaEmbedding)
        assert emb.provider_name == "ollama"
        assert emb.dimensions == 768

    def test_embed_empty_input(self):
        settings = EmbeddingSettings(provider="ollama", model="nomic", dimensions=768)
        emb = OllamaEmbedding(settings)
        assert emb.embed([]) == []


class TestQwenEmbedding:
    """Test Qwen (DashScope) Embedding with mocked API."""

    def test_factory_creates_qwen(self):
        settings = EmbeddingSettings(provider="qwen", model="text-embedding-v3", api_key="sk-test", dimensions=1024)
        emb = EmbeddingFactory.create(settings)
        assert isinstance(emb, QwenEmbedding)
        assert emb.provider_name == "qwen"

    def test_embed_empty_input(self):
        settings = EmbeddingSettings(provider="qwen", model="text-embedding-v3", api_key="sk-test")
        assert QwenEmbedding(settings).embed([]) == []

    @patch("openai.OpenAI")
    def test_embed_batches_over_ten(self, mock_openai_cls):
        """25 inputs must be split into 3 calls (10 + 10 + 5), order preserved."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        def fake_create(model, input, dimensions):
            # Echo one deterministic vector per input item.
            return MagicMock(data=[MagicMock(embedding=[float(len(t))]) for t in input])

        mock_client.embeddings.create.side_effect = fake_create

        settings = EmbeddingSettings(provider="qwen", model="text-embedding-v3", api_key="sk-test", dimensions=1)
        emb = QwenEmbedding(settings)
        texts = [f"{'x' * i}" for i in range(25)]  # lengths 0..24
        vectors = emb.embed(texts)

        # All 25 returned, in order, with batch size never exceeding 10.
        assert len(vectors) == 25
        assert vectors[0] == [0.0] and vectors[24] == [24.0]
        assert mock_client.embeddings.create.call_count == 3
        for call in mock_client.embeddings.create.call_args_list:
            assert len(call.kwargs["input"]) <= QwenEmbedding.MAX_BATCH_SIZE

    @patch("openai.OpenAI")
    def test_embed_single_batch_when_small(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=[0.1]), MagicMock(embedding=[0.2])]
        )
        settings = EmbeddingSettings(provider="qwen", model="text-embedding-v3", api_key="sk-test", dimensions=1)
        emb = QwenEmbedding(settings)
        vectors = emb.embed(["a", "b"])
        assert vectors == [[0.1], [0.2]]
        assert mock_client.embeddings.create.call_count == 1


class TestAllProvidersRegistered:
    """Verify all embedding providers are registered."""

    def test_all_registered(self):
        assert "openai" in _EMBEDDING_REGISTRY
        assert "azure" in _EMBEDDING_REGISTRY
        assert "ollama" in _EMBEDDING_REGISTRY
        assert "qwen" in _EMBEDDING_REGISTRY
