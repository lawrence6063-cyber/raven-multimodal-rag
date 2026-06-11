"""SplitterFactory — creates Splitter instances based on configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.libs.splitter.base_splitter import BaseSplitter, SplitterError

if TYPE_CHECKING:
    from src.core.settings import SplitterSettings


_SPLITTER_REGISTRY: dict[str, type[BaseSplitter]] = {}

# _builtins_registered 标记内置 splitter 实现是否已惰性注册
_builtins_registered = False


def register_splitter(provider: str):
    """Decorator to register a splitter implementation."""
    def decorator(cls: type[BaseSplitter]):
        _SPLITTER_REGISTRY[provider.lower()] = cls
        return cls
    return decorator


def _ensure_builtins_registered() -> None:
    """Register built-in splitter providers on demand (idempotent).

    Production entry points (e.g. scripts/ingest.py) construct a splitter via
    the factory without importing any implementation module first. We import the
    concrete classes explicitly and ``setdefault`` them so registration is robust
    even when modules are already cached in ``sys.modules`` (re-import would not
    re-run the decorator) and never overrides caller-registered providers.
    """
    global _builtins_registered
    if _builtins_registered:
        return
    _builtins_registered = True

    try:
        from src.libs.splitter.recursive_splitter import RecursiveSplitter
        _SPLITTER_REGISTRY.setdefault("recursive", RecursiveSplitter)
    except Exception:  # pragma: no cover - optional backend import failure
        pass
    try:
        from src.libs.splitter.semantic_splitter import SemanticSplitter
        _SPLITTER_REGISTRY.setdefault("semantic", SemanticSplitter)
    except Exception:  # pragma: no cover
        pass
    try:
        from src.libs.splitter.code_splitter import CodeSplitter
        _SPLITTER_REGISTRY.setdefault("code", CodeSplitter)
    except Exception:  # pragma: no cover
        pass
    try:
        from src.libs.splitter.document_structure_splitter import DocumentStructureSplitter
        _SPLITTER_REGISTRY.setdefault("document_structure", DocumentStructureSplitter)
    except Exception:  # pragma: no cover
        pass


class SplitterFactory:
    """Factory for creating Splitter instances based on settings."""

    @staticmethod
    def create(settings: "SplitterSettings") -> BaseSplitter:
        """Create a Splitter instance based on provider in settings.

        Args:
            settings: SplitterSettings with provider and configuration.

        Returns:
            An instance of BaseSplitter.

        Raises:
            SplitterError: If the provider is unknown.
        """
        _ensure_builtins_registered()
        provider = settings.provider.lower()

        if provider not in _SPLITTER_REGISTRY:
            available = ", ".join(sorted(_SPLITTER_REGISTRY.keys())) or "(none registered)"
            raise SplitterError(
                f"Unknown splitter provider: '{provider}'. Available: {available}",
                provider=provider,
            )

        cls = _SPLITTER_REGISTRY[provider]
        return cls(settings)

    @staticmethod
    def available_providers() -> list[str]:
        """Return list of registered provider names."""
        _ensure_builtins_registered()
        return sorted(_SPLITTER_REGISTRY.keys())
