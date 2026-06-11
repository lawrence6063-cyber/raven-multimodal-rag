"""BaseTransform — abstract interface for chunk transformation steps."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.core.types import Chunk


class BaseTransform(ABC):
    """Abstract base class for chunk transformation steps.

    Transforms are applied sequentially after chunking to enrich/clean chunks.
    """

    @abstractmethod
    def transform(self, chunks: list[Chunk]) -> list[Chunk]:
        """Transform a list of chunks.

        Args:
            chunks: List of Chunk objects to transform.

        Returns:
            Transformed list of Chunk objects.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the transform step name."""
