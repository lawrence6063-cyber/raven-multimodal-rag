"""VisionLLMFactory — creates Vision LLM instances based on configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.libs.llm.base_vision_llm import BaseVisionLLM
from src.libs.llm.base_llm import LLMError

if TYPE_CHECKING:
    from src.core.settings import VisionLLMSettings


# Registry of provider name -> class (populated by implementations)
_VISION_LLM_REGISTRY: dict[str, type[BaseVisionLLM]] = {}

# _builtins_registered 标记内置 Vision LLM 实现是否已惰性导入注册
_builtins_registered = False


def register_vision_llm(provider: str):
    """Decorator to register a Vision LLM provider implementation."""
    def decorator(cls: type[BaseVisionLLM]):
        _VISION_LLM_REGISTRY[provider.lower()] = cls
        return cls
    return decorator


def _ensure_builtins_registered() -> None:
    """Register built-in Vision LLM providers on demand (idempotent).

    Mirrors :func:`src.libs.llm.llm_factory._ensure_builtins_registered`: import
    concrete classes explicitly and ``setdefault`` them so registration is robust
    even when modules are already cached in ``sys.modules`` (re-import would not
    re-run the decorator), while never overriding caller/test registrations.
    """
    global _builtins_registered
    if _builtins_registered:
        return
    _builtins_registered = True

    try:
        from src.libs.llm.azure_vision_llm import AzureVisionLLM
        _VISION_LLM_REGISTRY.setdefault("azure_vision", AzureVisionLLM)
    except Exception:  # pragma: no cover - optional backend import failure
        pass
    try:
        from src.libs.llm.qwen_vision_llm import QwenVisionLLM
        _VISION_LLM_REGISTRY.setdefault("qwen_vision", QwenVisionLLM)
    except Exception:  # pragma: no cover
        pass


class VisionLLMFactory:
    """Factory for creating Vision LLM instances based on settings."""

    @staticmethod
    def create(settings: "VisionLLMSettings") -> BaseVisionLLM:
        """Create a Vision LLM instance based on the provider in settings.

        Args:
            settings: VisionLLMSettings with provider and configuration.

        Returns:
            An instance of BaseVisionLLM.

        Raises:
            LLMError: If the provider is unknown.
        """
        _ensure_builtins_registered()
        provider = (settings.provider or "").lower()

        if provider not in _VISION_LLM_REGISTRY:
            available = ", ".join(sorted(_VISION_LLM_REGISTRY.keys())) or "(none registered)"
            raise LLMError(
                f"Unknown vision LLM provider: '{provider}'. Available: {available}",
                provider=provider,
            )

        cls = _VISION_LLM_REGISTRY[provider]
        return cls(settings)

    @staticmethod
    def available_providers() -> list[str]:
        """Return list of registered provider names."""
        _ensure_builtins_registered()
        return sorted(_VISION_LLM_REGISTRY.keys())
