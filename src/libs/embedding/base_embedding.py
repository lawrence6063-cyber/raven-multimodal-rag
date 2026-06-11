"""BaseEmbedding — abstract interface for all embedding providers."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseEmbedding(ABC):
    """Abstract base class for embedding providers.

    All embedding implementations must subclass this and implement embed.
    """

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts into vectors.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors (each is a list of floats).

        Raises:
            EmbeddingError: If the API call fails.
        """

    def embed_image(self, images: list[str | bytes]) -> list[list[float]]:
        """Embed a list of images into vectors (multimodal providers only).

        Text-only providers do not support this and raise ``NotImplementedError``;
        multimodal providers (e.g. DashScope multimodal embedding) override it to
        place images into the *same* vector space as text, enabling cross-modal
        retrieval.

        Args:
            images: List of images, each given as a local file path, raw bytes,
                or a base64 data URI.

        Returns:
            List of embedding vectors aligned with the input order.

        Raises:
            NotImplementedError: If the provider has no image support.
            EmbeddingError: If the API call fails.
        """
        raise NotImplementedError(
            f"{self.provider_name} does not support image embedding"
        )

    def supports_images(self) -> bool:
        """Whether this provider can embed images into the text vector space."""
        return type(self).embed_image is not BaseEmbedding.embed_image

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name."""

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Return the embedding vector dimensions."""


class EmbeddingError(Exception):
    """Raised when an embedding API call fails."""

    def __init__(self, message: str, provider: str = "", cause: Exception | None = None):
        self.provider = provider
        self.cause = cause
        super().__init__(f"[{provider}] {message}" if provider else message)
