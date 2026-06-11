"""BaseReranker — abstract interface for reranking strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RerankCandidate:
    """A candidate document for reranking."""

    id: str
    text: str
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseReranker(ABC):
    """Abstract base class for reranker implementations."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
    ) -> list[RerankCandidate]:
        """Rerank candidates by relevance to the query.

        Args:
            query: The search query.
            candidates: List of candidates to rerank.

        Returns:
            Reranked list of candidates (most relevant first).

        Raises:
            RerankerError: If reranking fails.
        """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the reranker provider name."""


class NoneReranker(BaseReranker):
    """No-op reranker that preserves original ordering."""

    def rerank(self, query: str, candidates: list[RerankCandidate]) -> list[RerankCandidate]:
        """Return candidates unchanged."""
        return candidates

    @property
    def provider_name(self) -> str:
        return "none"


class RerankerError(Exception):
    """Raised when reranking fails."""

    def __init__(self, message: str, provider: str = ""):
        self.provider = provider
        super().__init__(f"[{provider}] {message}" if provider else message)
