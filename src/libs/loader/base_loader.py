"""BaseLoader — abstract interface for document loading."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.core.types import Document


class BaseLoader(ABC):
    """Abstract base class for document loaders."""

    @abstractmethod
    def load(self, path: str) -> Document:
        """Load a document from file path.

        Args:
            path: Path to the document file.

        Returns:
            Document with text content and metadata.

        Raises:
            LoaderError: If loading fails.
        """

    @property
    @abstractmethod
    def supported_extensions(self) -> list[str]:
        """Return list of supported file extensions (e.g., ['.pdf'])."""


class LoaderError(Exception):
    """Raised when document loading fails."""

    def __init__(self, message: str, path: str = ""):
        self.path = path
        super().__init__(f"[{path}] {message}" if path else message)
