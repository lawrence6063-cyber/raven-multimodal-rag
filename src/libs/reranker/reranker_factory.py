"""RerankerFactory — creates Reranker instances based on configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.libs.reranker.base_reranker import BaseReranker, NoneReranker, RerankerError

if TYPE_CHECKING:
    from src.core.settings import RerankSettings


_RERANKER_REGISTRY: dict[str, type[BaseReranker]] = {
    "none": NoneReranker,
}

# _builtins_registered 标记内置 reranker 实现是否已惰性注册
_builtins_registered = False


def register_reranker(provider: str):
    """Decorator to register a reranker implementation."""
    def decorator(cls: type[BaseReranker]):
        _RERANKER_REGISTRY[provider.lower()] = cls
        return cls
    return decorator


def _ensure_builtins_registered() -> None:
    """Register built-in reranker providers on demand (idempotent).

    ``none`` is always present; the optional cross-encoder / LLM rerankers are
    imported explicitly and ``setdefault`` so they register reliably from
    production entry points even when their modules are already cached.
    """
    global _builtins_registered
    if _builtins_registered:
        return
    _builtins_registered = True

    try:
        from src.libs.reranker.cross_encoder_reranker import CrossEncoderReranker
        _RERANKER_REGISTRY.setdefault("cross_encoder", CrossEncoderReranker)
    except Exception:  # pragma: no cover - optional backend import failure
        pass
    try:
        from src.libs.reranker.llm_reranker import LLMReranker
        _RERANKER_REGISTRY.setdefault("llm", LLMReranker)
    except Exception:  # pragma: no cover
        pass


class RerankerFactory:
    """Factory for creating Reranker instances based on settings."""

    @staticmethod
    def create(settings: "RerankSettings") -> BaseReranker:
        """Create a Reranker instance based on provider in settings.

        If rerank is disabled, returns NoneReranker regardless of provider.

        Args:
            settings: RerankSettings with provider and configuration.

        Returns:
            An instance of BaseReranker.

        Raises:
            RerankerError: If the provider is unknown.
        """
        if not settings.enabled:
            return NoneReranker()

        _ensure_builtins_registered()
        provider = settings.provider.lower()

        if provider not in _RERANKER_REGISTRY:
            available = ", ".join(sorted(_RERANKER_REGISTRY.keys()))
            raise RerankerError(
                f"Unknown reranker provider: '{provider}'. Available: {available}",
                provider=provider,
            )

        cls = _RERANKER_REGISTRY[provider]
        if cls == NoneReranker:
            return NoneReranker()
        return cls(settings)

    @staticmethod
    def available_providers() -> list[str]:
        """Return list of registered provider names."""
        _ensure_builtins_registered()
        return sorted(_RERANKER_REGISTRY.keys())
