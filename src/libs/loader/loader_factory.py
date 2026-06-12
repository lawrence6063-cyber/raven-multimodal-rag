"""LoaderFactory — creates document loader instances based on configuration.

Mirrors :mod:`src.libs.splitter.splitter_factory`: providers are registered via a
decorator and built-in implementations are registered lazily (idempotent) so the
factory works even when concrete modules are already cached in ``sys.modules``.

Built-in providers:
- ``markitdown``: the legacy :class:`PdfLoader` (default / fallback).
- ``pymupdf``: layout-aware :class:`PyMuPDFLoader` (de-hyphenation, two-column
  reading order, table extraction, figure-caption anchoring).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.libs.loader.base_loader import BaseLoader, LoaderError

if TYPE_CHECKING:
    from src.core.settings import LoaderSettings


# _LOADER_REGISTRY 注册表：provider 名 -> loader 类
_LOADER_REGISTRY: dict[str, type[BaseLoader]] = {}

# _builtins_registered 标记内置 loader 实现是否已惰性注册
_builtins_registered = False


def register_loader(provider: str):
    """Decorator to register a loader implementation under ``provider``."""

    def decorator(cls: type[BaseLoader]):
        _LOADER_REGISTRY[provider.lower()] = cls
        return cls

    return decorator


def _ensure_builtins_registered() -> None:
    """Register built-in loader providers on demand (idempotent)."""
    global _builtins_registered
    if _builtins_registered:
        return
    _builtins_registered = True

    try:
        from src.libs.loader.pdf_loader import PdfLoader

        _LOADER_REGISTRY.setdefault("markitdown", PdfLoader)
    except Exception:  # pragma: no cover - optional backend import failure
        pass
    try:
        from src.libs.loader.pymupdf_loader import PyMuPDFLoader

        _LOADER_REGISTRY.setdefault("pymupdf", PyMuPDFLoader)
    except Exception:  # pragma: no cover
        pass


class LoaderFactory:
    """Factory for creating document loader instances based on settings."""

    @staticmethod
    def create(settings: "LoaderSettings") -> BaseLoader:
        """Create a loader instance based on the configured provider.

        Args:
            settings: LoaderSettings with provider and configuration.

        Returns:
            An instance of BaseLoader.

        Raises:
            LoaderError: If the provider is unknown.
        """
        _ensure_builtins_registered()
        provider = settings.provider.lower()

        if provider not in _LOADER_REGISTRY:
            available = ", ".join(sorted(_LOADER_REGISTRY.keys())) or "(none registered)"
            raise LoaderError(
                f"Unknown loader provider: '{provider}'. Available: {available}"
            )

        cls = _LOADER_REGISTRY[provider]
        return cls(settings)

    @staticmethod
    def available_providers() -> list[str]:
        """Return list of registered provider names."""
        _ensure_builtins_registered()
        return sorted(_LOADER_REGISTRY.keys())
