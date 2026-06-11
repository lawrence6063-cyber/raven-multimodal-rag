"""DenseEncoder — batch encodes chunks into dense vectors."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.core.types import Chunk, ChunkRecord
from src.libs.embedding.embedding_factory import EmbeddingFactory

if TYPE_CHECKING:
    from src.core.settings import Settings


class DenseEncoder:
    """Encodes chunk texts into dense embedding vectors."""

    def __init__(self, settings: "Settings"):
        self._embedding = EmbeddingFactory.create(settings.embedding)

    def encode(self, chunks: list[Chunk]) -> list[ChunkRecord]:
        """Encode chunks into ChunkRecords with dense vectors.

        Args:
            chunks: List of Chunk objects.

        Returns:
            List of ChunkRecord objects with dense_vector populated.
        """
        if not chunks:
            return []

        texts = [c.text for c in chunks]
        vectors = self._embedding.embed(texts)

        records = []
        for chunk, vector in zip(chunks, vectors):
            records.append(ChunkRecord(
                id=chunk.id,
                text=chunk.text,
                metadata=chunk.metadata,
                dense_vector=vector,
            ))

        return records
