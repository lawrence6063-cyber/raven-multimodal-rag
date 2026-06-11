"""VectorStoreFactory — creates VectorStore instances based on configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.libs.vector_store.base_vector_store import BaseVectorStore, VectorStoreError

if TYPE_CHECKING:
    from src.core.settings import VectorStoreSettings


_VECTOR_STORE_REGISTRY: dict[str, type[BaseVectorStore]] = {}


def register_vector_store(provider: str):
    """Decorator to register a vector store implementation."""
    def decorator(cls: type[BaseVectorStore]):
        _VECTOR_STORE_REGISTRY[provider.lower()] = cls
        return cls
    return decorator


def _ensure_builtins_registered() -> None:
    """Import built-in implementations so their decorators self-register.

    Production entry points (e.g. the MCP server) may construct a store without
    having imported any concrete implementation module first; this lazily wires
    the bundled providers on demand.
    """
    if _VECTOR_STORE_REGISTRY:
        return
    try:
        from src.libs.vector_store import chroma_store  # noqa: F401  registers "chroma"
    except Exception:  # pragma: no cover - optional backend import failure
        pass


class VectorStoreFactory:
    """Factory for creating VectorStore instances based on settings."""

    @staticmethod
    def create(settings: "VectorStoreSettings") -> BaseVectorStore:
        """Create a VectorStore instance based on provider in settings.

        Args:
            settings: VectorStoreSettings with provider and configuration.

        Returns:
            An instance of BaseVectorStore.

        Raises:
            VectorStoreError: If the provider is unknown.
        """
        provider = settings.provider.lower()

        _ensure_builtins_registered()

        if provider not in _VECTOR_STORE_REGISTRY:
            available = ", ".join(sorted(_VECTOR_STORE_REGISTRY.keys())) or "(none registered)"
            raise VectorStoreError(
                f"Unknown vector store provider: '{provider}'. Available: {available}",
                provider=provider,
            )

        cls = _VECTOR_STORE_REGISTRY[provider]
        return cls(settings)

    @staticmethod
    def available_providers() -> list[str]:
        """Return list of registered provider names."""
        return sorted(_VECTOR_STORE_REGISTRY.keys())
