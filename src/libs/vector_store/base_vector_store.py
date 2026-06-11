"""BaseVectorStore — abstract interface for vector database operations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class VectorRecord:
    """A record to be stored in the vector database."""

    id: str
    vector: list[float]
    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryResult:
    """A single result from a vector query."""

    id: str
    score: float
    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseVectorStore(ABC):
    """Abstract base class for vector store implementations."""

    @abstractmethod
    def upsert(self, records: list[VectorRecord]) -> None:
        """Upsert records into the vector store.

        Args:
            records: List of VectorRecord objects to store.

        Raises:
            VectorStoreError: If the operation fails.
        """

    @abstractmethod
    def query(
        self,
        vector: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[QueryResult]:
        """Query the vector store for similar vectors.

        Args:
            vector: Query vector.
            top_k: Number of results to return.
            filters: Optional metadata filters.

        Returns:
            List of QueryResult sorted by similarity (highest first).

        Raises:
            VectorStoreError: If the query fails.
        """

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        """Delete records by ID.

        Args:
            ids: List of record IDs to delete.
        """

    @abstractmethod
    def get_by_ids(self, ids: list[str]) -> list["QueryResult"]:
        """Retrieve records by their IDs (batch fetch).

        用于按 ID 批量获取记录的文本和元数据，不涉及相似度计算。

        Args:
            ids: List of record IDs to fetch.

        Returns:
            List of QueryResult for found IDs (score=0.0).
            IDs not found in the store are silently skipped.
            Returns empty list if ids is empty.

        Raises:
            VectorStoreError: If the operation fails.
        """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the vector store provider name."""


class VectorStoreError(Exception):
    """Raised when a vector store operation fails."""

    def __init__(self, message: str, provider: str = ""):
        self.provider = provider
        super().__init__(f"[{provider}] {message}" if provider else message)
