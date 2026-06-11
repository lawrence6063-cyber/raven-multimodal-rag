"""EmbeddingFactory — creates Embedding instances based on configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.libs.embedding.base_embedding import BaseEmbedding, EmbeddingError

if TYPE_CHECKING:
    from src.core.settings import EmbeddingSettings


_EMBEDDING_REGISTRY: dict[str, type[BaseEmbedding]] = {}


def register_embedding(provider: str):
    """Decorator to register an embedding provider implementation."""
    def decorator(cls: type[BaseEmbedding]):
        _EMBEDDING_REGISTRY[provider.lower()] = cls
        return cls
    return decorator


def _ensure_builtins_registered() -> None:
    """Import built-in implementations so their decorators self-register.

    Production entry points (e.g. query/evaluate scripts) may construct an
    embedding client without having imported any concrete implementation module
    first; this lazily wires the bundled providers on demand.
    """
    if _EMBEDDING_REGISTRY:
        return
    for module in (
        "azure_embedding",
        "openai_embedding",
        "ollama_embedding",
        "qwen_embedding",
        "qwen_multimodal_embedding",
    ):
        try:
            __import__(f"src.libs.embedding.{module}")
        except Exception:  # pragma: no cover - optional backend import failure
            pass


class EmbeddingFactory:
    """Factory for creating Embedding instances based on settings."""

    @staticmethod
    def create(settings: "EmbeddingSettings") -> BaseEmbedding:
        """Create an Embedding instance based on provider in settings.

        Args:
            settings: EmbeddingSettings with provider and configuration.

        Returns:
            An instance of BaseEmbedding.

        Raises:
            EmbeddingError: If the provider is unknown.
        """
        provider = settings.provider.lower()

        _ensure_builtins_registered()

        if provider not in _EMBEDDING_REGISTRY:
            available = ", ".join(sorted(_EMBEDDING_REGISTRY.keys())) or "(none registered)"
            raise EmbeddingError(
                f"Unknown embedding provider: '{provider}'. Available: {available}",
                provider=provider,
            )

        cls = _EMBEDDING_REGISTRY[provider]
        return cls(settings)

    @staticmethod
    def available_providers() -> list[str]:
        """Return list of registered provider names."""
        _ensure_builtins_registered()
        return sorted(_EMBEDDING_REGISTRY.keys())
