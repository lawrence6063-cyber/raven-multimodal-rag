"""Azure OpenAI Embedding implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.libs.embedding.base_embedding import BaseEmbedding, EmbeddingError
from src.libs.embedding.embedding_factory import register_embedding

if TYPE_CHECKING:
    from src.core.settings import EmbeddingSettings


@register_embedding("azure")
class AzureEmbedding(BaseEmbedding):
    """Azure OpenAI Embedding implementation."""

    def __init__(self, settings: "EmbeddingSettings"):
        self._settings = settings
        self._model = settings.deployment_name or settings.model
        self._api_key = settings.api_key
        self._azure_endpoint = settings.azure_endpoint
        self._api_version = settings.api_version
        self._dimensions = settings.dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using Azure OpenAI API."""
        if not texts:
            return []

        import openai

        try:
            client = openai.AzureOpenAI(
                api_key=self._api_key,
                azure_endpoint=self._azure_endpoint,
                api_version=self._api_version,
            )
            response = client.embeddings.create(model=self._model, input=texts)
            return [item.embedding for item in response.data]
        except openai.AuthenticationError as e:
            raise EmbeddingError("Azure authentication failed.", provider="azure", cause=e) from e
        except Exception as e:
            raise EmbeddingError(f"Azure Embedding failed: {e}", provider="azure", cause=e) from e

    @property
    def provider_name(self) -> str:
        return "azure"

    @property
    def dimensions(self) -> int:
        return self._dimensions
