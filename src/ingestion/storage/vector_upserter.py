"""VectorUpserter — upserts ChunkRecords to VectorStore with idempotency."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.core.types import ChunkRecord
from src.libs.vector_store.base_vector_store import VectorRecord
from src.libs.vector_store.vector_store_factory import VectorStoreFactory

if TYPE_CHECKING:
    from src.core.settings import Settings


class VectorUpserter:
    """Upserts ChunkRecords to the configured vector store."""

    def __init__(self, settings: "Settings"):
        self._store = VectorStoreFactory.create(settings.vector_store)

    def upsert(self, records: list[ChunkRecord]) -> None:
        """Upsert records to vector store (idempotent via stable IDs).

        Args:
            records: List of ChunkRecord with dense_vector populated.
        """
        if not records:
            return

        vector_records = [
            VectorRecord(
                id=rec.id,
                vector=rec.dense_vector,
                text=rec.text,
                metadata=rec.metadata,
            )
            for rec in records
            if rec.dense_vector  # Skip records without vectors
        ]

        if vector_records:
            self._store.upsert(vector_records)
