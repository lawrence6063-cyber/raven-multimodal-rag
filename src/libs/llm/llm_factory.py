"""LLMFactory — creates LLM instances based on configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.libs.llm.base_llm import BaseLLM, LLMError

if TYPE_CHECKING:
    from src.core.settings import LLMSettings


# Registry of provider name -> class (populated by implementations)
_LLM_REGISTRY: dict[str, type[BaseLLM]] = {}

# _builtins_registered 标记内置 LLM 实现是否已惰性导入注册
_builtins_registered = False


def register_llm(provider: str):
    """Decorator to register an LLM provider implementation."""
    def decorator(cls: type[BaseLLM]):
        _LLM_REGISTRY[provider.lower()] = cls
        return cls
    return decorator


def _ensure_builtins_registered() -> None:
    """Register built-in LLM providers on demand (idempotent).

    Registration normally happens via the ``@register_llm`` decorator at import
    time. Production entry points (query/evaluate scripts, MCP server) may
    construct an LLM without importing any concrete implementation module first.

    We import the concrete classes explicitly and ``setdefault`` them into the
    registry: this is robust even when a module is already cached in
    ``sys.modules`` (in which case re-importing would NOT re-run the decorator),
    while never overriding a provider that callers/tests registered themselves.
    """
    global _builtins_registered
    if _builtins_registered:
        return
    _builtins_registered = True

    try:
        from src.libs.llm.openai_llm import OpenAILLM
        _LLM_REGISTRY.setdefault("openai", OpenAILLM)
    except Exception:  # pragma: no cover - optional backend import failure
        pass
    try:
        from src.libs.llm.azure_llm import AzureLLM
        _LLM_REGISTRY.setdefault("azure", AzureLLM)
    except Exception:  # pragma: no cover
        pass
    try:
        from src.libs.llm.ollama_llm import OllamaLLM
        _LLM_REGISTRY.setdefault("ollama", OllamaLLM)
    except Exception:  # pragma: no cover
        pass
    try:
        from src.libs.llm.deepseek_llm import DeepSeekLLM
        _LLM_REGISTRY.setdefault("deepseek", DeepSeekLLM)
    except Exception:  # pragma: no cover
        pass
    try:
        from src.libs.llm.qwen_llm import QwenLLM
        _LLM_REGISTRY.setdefault("qwen", QwenLLM)
    except Exception:  # pragma: no cover
        pass


class LLMFactory:
    """Factory for creating LLM instances based on settings."""

    @staticmethod
    def create(settings: "LLMSettings") -> BaseLLM:
        """Create an LLM instance based on the provider in settings.

        Args:
            settings: LLMSettings with provider and configuration.

        Returns:
            An instance of BaseLLM.

        Raises:
            LLMError: If the provider is unknown or creation fails.
        """
        _ensure_builtins_registered()
        provider = settings.provider.lower()

        if provider not in _LLM_REGISTRY:
            available = ", ".join(sorted(_LLM_REGISTRY.keys())) or "(none registered)"
            raise LLMError(
                f"Unknown LLM provider: '{provider}'. Available: {available}",
                provider=provider,
            )

        cls = _LLM_REGISTRY[provider]
        return cls(settings)

    @staticmethod
    def available_providers() -> list[str]:
        """Return list of registered provider names."""
        _ensure_builtins_registered()
        return sorted(_LLM_REGISTRY.keys())
