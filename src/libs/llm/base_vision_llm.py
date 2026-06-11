"""BaseVisionLLM — abstract interface for vision-capable LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from src.libs.llm.base_llm import ChatResponse, LLMError

if TYPE_CHECKING:
    pass


class BaseVisionLLM(ABC):
    """Abstract base class for Vision LLM providers.

    Supports multimodal input (text + image) for image understanding tasks.
    """

    @abstractmethod
    def chat_with_image(self, text: str, image_path: str | bytes) -> ChatResponse:
        """Send text + image to the Vision LLM.

        Args:
            text: Text prompt describing what to analyze.
            image_path: Path to image file, or raw bytes of the image.

        Returns:
            ChatResponse with the model's description/analysis.

        Raises:
            LLMError: If the API call fails.
        """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the vision LLM provider name."""
