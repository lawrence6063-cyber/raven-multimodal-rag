"""OpenAI Embedding implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.libs.embedding.base_embedding import BaseEmbedding, EmbeddingError
from src.libs.embedding.embedding_factory import register_embedding

if TYPE_CHECKING:
    from src.core.settings import EmbeddingSettings


@register_embedding("openai")
class OpenAIEmbedding(BaseEmbedding):
    """OpenAI Embedding API implementation."""

    DEFAULT_BASE_URL = "https://api.openai.com/v1"

    def __init__(self, settings: "EmbeddingSettings"):
        self._settings = settings
        self._model = settings.model
        self._api_key = settings.api_key
        self._base_url = settings.base_url or self.DEFAULT_BASE_URL
        self._dimensions = settings.dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using OpenAI API."""
        if not texts:
            return []

        import openai

        try:
            client = openai.OpenAI(api_key=self._api_key, base_url=self._base_url)
            response = client.embeddings.create(model=self._model, input=texts)
            return [item.embedding for item in response.data]
        except openai.AuthenticationError as e:
            raise EmbeddingError("Authentication failed.", provider="openai", cause=e) from e
        except Exception as e:
            raise EmbeddingError(f"Embedding API failed: {e}", provider="openai", cause=e) from e

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def dimensions(self) -> int:
        return self._dimensions
