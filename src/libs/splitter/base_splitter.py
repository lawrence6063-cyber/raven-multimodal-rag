"""BaseSplitter — abstract interface for text splitting strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseSplitter(ABC):
    """Abstract base class for text splitters.

    All splitter implementations must subclass this and implement split_text.
    """

    @abstractmethod
    def split_text(self, text: str) -> list[str]:
        """Split text into chunks.

        Args:
            text: The text to split.

        Returns:
            List of text chunks.

        Raises:
            SplitterError: If splitting fails.
        """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the splitter strategy name."""


class SplitterError(Exception):
    """Raised when text splitting fails."""

    def __init__(self, message: str, provider: str = ""):
        self.provider = provider
        super().__init__(f"[{provider}] {message}" if provider else message)
