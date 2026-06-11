"""Ollama Embedding implementation — local HTTP endpoint."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.libs.embedding.base_embedding import BaseEmbedding, EmbeddingError
from src.libs.embedding.embedding_factory import register_embedding

if TYPE_CHECKING:
    from src.core.settings import EmbeddingSettings


@register_embedding("ollama")
class OllamaEmbedding(BaseEmbedding):
    """Ollama local embedding implementation via HTTP API."""

    DEFAULT_BASE_URL = "http://localhost:11434"

    def __init__(self, settings: "EmbeddingSettings"):
        self._settings = settings
        self._model = settings.model
        self._base_url = settings.base_url or self.DEFAULT_BASE_URL
        self._dimensions = settings.dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using Ollama HTTP API."""
        if not texts:
            return []

        import urllib.request
        import urllib.error
        import json

        url = f"{self._base_url.rstrip('/')}/api/embed"
        results = []

        try:
            # Ollama supports batch embedding via /api/embed
            payload = {"model": self._model, "input": texts}
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read().decode("utf-8"))

            embeddings = body.get("embeddings", [])
            if len(embeddings) == len(texts):
                return embeddings

            # Fallback: single request per text
            for text in texts:
                payload = {"model": self._model, "input": [text]}
                data = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=60) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                embs = body.get("embeddings", [[]])
                results.append(embs[0] if embs else [0.0] * self._dimensions)

            return results

        except urllib.error.URLError as e:
            raise EmbeddingError(
                f"Cannot connect to Ollama at {self._base_url}. Is it running?",
                provider="ollama",
                cause=e,
            ) from e
        except Exception as e:
            raise EmbeddingError(f"Ollama embedding failed: {e}", provider="ollama", cause=e) from e

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def dimensions(self) -> int:
        return self._dimensions
