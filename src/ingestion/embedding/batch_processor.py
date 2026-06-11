"""BatchProcessor — processes chunks in batches for encoding."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.core.types import Chunk, ChunkRecord
from src.ingestion.embedding.dense_encoder import DenseEncoder
from src.ingestion.embedding.sparse_encoder import SparseEncoder

if TYPE_CHECKING:
    from src.core.settings import Settings


class BatchProcessor:
    """Processes chunks in batches, driving dense and sparse encoding."""

    def __init__(self, settings: "Settings"):
        self._batch_size = settings.ingestion.batch_size
        self._dense_encoder = DenseEncoder(settings)
        self._sparse_encoder = SparseEncoder()

    def process(self, chunks: list[Chunk]) -> list[ChunkRecord]:
        """Encode chunks in batches, merging dense and sparse vectors.

        Args:
            chunks: All chunks to encode.

        Returns:
            List of ChunkRecord with both dense_vector and sparse_vector.
        """
        if not chunks:
            return []

        all_records: list[ChunkRecord] = []

        for batch_start in range(0, len(chunks), self._batch_size):
            batch = chunks[batch_start:batch_start + self._batch_size]

            # Dense encoding (API call)
            dense_records = self._dense_encoder.encode(batch)
            # Sparse encoding (local computation)
            sparse_records = self._sparse_encoder.encode(batch)

            # Merge dense + sparse into single records
            for dense_rec, sparse_rec in zip(dense_records, sparse_records):
                merged = ChunkRecord(
                    id=dense_rec.id,
                    text=dense_rec.text,
                    metadata=dense_rec.metadata,
                    dense_vector=dense_rec.dense_vector,
                    sparse_vector=sparse_rec.sparse_vector,
                )
                all_records.append(merged)

        return all_records
