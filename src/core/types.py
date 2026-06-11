"""Core data types — shared contracts for ingestion, retrieval, and MCP tools.

Defines Document, Chunk, ChunkRecord as the canonical data structures
flowing through the entire pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """A parsed document ready for chunking.

    Produced by Loaders, consumed by Splitters/Transforms.
    """

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {"id": self.id, "text": self.text, "metadata": self.metadata}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Document":
        """Deserialize from dict."""
        return cls(id=data["id"], text=data["text"], metadata=data.get("metadata", {}))


@dataclass
class Chunk:
    """A text chunk derived from a Document.

    Produced by Splitters, enriched by Transforms, consumed by Encoders.
    """

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    source_ref: str = ""  # Parent Document.id

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata,
            "source_ref": self.source_ref,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chunk":
        """Deserialize from dict."""
        return cls(
            id=data["id"],
            text=data["text"],
            metadata=data.get("metadata", {}),
            source_ref=data.get("source_ref", ""),
        )


@dataclass
class ChunkRecord:
    """A chunk with computed vectors, ready for storage.

    Produced by Encoders, consumed by VectorUpserter/BM25Indexer.
    """

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    dense_vector: list[float] = field(default_factory=list)
    sparse_vector: dict[str, float] = field(default_factory=dict)  # term -> weight

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata,
            "dense_vector": self.dense_vector,
            "sparse_vector": self.sparse_vector,
        }


@dataclass
class RetrievalResult:
    """A single retrieval result from search.

    Used by DenseRetriever, SparseRetriever, Fusion, and Reranker.
    """

    chunk_id: str
    score: float
    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
