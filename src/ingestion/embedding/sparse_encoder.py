"""SparseEncoder — computes BM25 term statistics for chunks."""

from __future__ import annotations

import math
import re
from collections import Counter

from src.core.types import Chunk, ChunkRecord


class SparseEncoder:
    """Computes term frequency weights for BM25-style sparse retrieval."""

    # Common English stop words
    STOP_WORDS = frozenset([
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "dare", "ought",
        "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "through", "during", "before", "after", "above", "below",
        "between", "out", "off", "over", "under", "again", "further", "then",
        "once", "here", "there", "when", "where", "why", "how", "all", "each",
        "every", "both", "few", "more", "most", "other", "some", "such", "no",
        "nor", "not", "only", "own", "same", "so", "than", "too", "very",
        "and", "but", "or", "if", "this", "that", "these", "those", "it",
    ])

    def encode(self, chunks: list[Chunk]) -> list[ChunkRecord]:
        """Compute sparse (term weight) vectors for chunks.

        Args:
            chunks: List of Chunk objects.

        Returns:
            List of ChunkRecord with sparse_vector populated (term -> tf weight).
        """
        if not chunks:
            return []

        records = []
        for chunk in chunks:
            terms = self._tokenize(chunk.text)
            tf = Counter(terms)
            # Normalize TF by document length
            doc_len = len(terms) if terms else 1
            sparse_vector = {term: count / doc_len for term, count in tf.items()}

            records.append(ChunkRecord(
                id=chunk.id,
                text=chunk.text,
                metadata=chunk.metadata,
                sparse_vector=sparse_vector,
            ))

        return records

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into lowercase terms, removing stop words."""
        # Simple whitespace + punctuation tokenization
        tokens = re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())
        return [t for t in tokens if t not in self.STOP_WORDS and len(t) > 1]
