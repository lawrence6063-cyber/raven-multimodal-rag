"""BM25Indexer — builds and persists inverted index with IDF statistics."""

from __future__ import annotations

import json
import math
import pickle
from pathlib import Path
from collections import defaultdict

from src.core.types import ChunkRecord


class BM25Indexer:
    """Builds BM25 inverted index and supports querying."""

    def __init__(self, index_path: str = "data/db/bm25"):
        self._index_path = Path(index_path)
        self._index_path.mkdir(parents=True, exist_ok=True)
        self._inverted_index: dict[str, list[dict]] = {}  # term -> [{chunk_id, tf, doc_length}]
        self._idf: dict[str, float] = {}
        self._doc_count: int = 0
        self._avg_doc_length: float = 0.0
        self._loaded = False

    def build(self, records: list[ChunkRecord]) -> None:
        """Build inverted index from ChunkRecords with sparse vectors."""
        if not records:
            return

        inverted: dict[str, list[dict]] = defaultdict(list)
        doc_lengths = []

        for rec in records:
            doc_len = sum(rec.sparse_vector.values()) if rec.sparse_vector else 0
            doc_lengths.append(doc_len)
            for term, tf in rec.sparse_vector.items():
                inverted[term].append({"chunk_id": rec.id, "tf": tf, "doc_length": doc_len})

        self._inverted_index = dict(inverted)
        self._doc_count = len(records)
        self._avg_doc_length = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0

        # Compute IDF
        for term, postings in self._inverted_index.items():
            df = len(postings)
            self._idf[term] = math.log((self._doc_count - df + 0.5) / (df + 0.5))

        self._save()
        self._loaded = True

    def query(self, keywords: list[str], top_k: int = 10, k1: float = 1.5, b: float = 0.75) -> list[dict]:
        """Query the BM25 index.

        Args:
            keywords: Query terms.
            top_k: Number of results.
            k1: BM25 term frequency saturation parameter.
            b: BM25 length normalization parameter.

        Returns:
            List of {"chunk_id": str, "score": float} sorted by score descending.
        """
        if not self._loaded:
            self._load()

        scores: dict[str, float] = defaultdict(float)

        for term in keywords:
            term_lower = term.lower()
            if term_lower not in self._inverted_index:
                continue
            idf = self._idf.get(term_lower, 0.0)
            for posting in self._inverted_index[term_lower]:
                tf = posting["tf"]
                dl = posting["doc_length"]
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * dl / self._avg_doc_length) if self._avg_doc_length > 0 else tf + k1
                score = idf * (numerator / denominator)
                scores[posting["chunk_id"]] += score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [{"chunk_id": cid, "score": score} for cid, score in ranked]

    def _save(self) -> None:
        """Persist index to disk."""
        data = {
            "inverted_index": self._inverted_index,
            "idf": self._idf,
            "doc_count": self._doc_count,
            "avg_doc_length": self._avg_doc_length,
        }
        with open(self._index_path / "bm25_index.pkl", "wb") as f:
            pickle.dump(data, f)

    def _load(self) -> None:
        """Load index from disk."""
        index_file = self._index_path / "bm25_index.pkl"
        if index_file.exists():
            with open(index_file, "rb") as f:
                data = pickle.load(f)
            self._inverted_index = data["inverted_index"]
            self._idf = data["idf"]
            self._doc_count = data["doc_count"]
            self._avg_doc_length = data["avg_doc_length"]
        self._loaded = True

    def remove_document(self, chunk_ids: list[str]) -> None:
        """Remove chunks from the index by their IDs."""
        ids_set = set(chunk_ids)
        for term in list(self._inverted_index.keys()):
            self._inverted_index[term] = [p for p in self._inverted_index[term] if p["chunk_id"] not in ids_set]
            if not self._inverted_index[term]:
                del self._inverted_index[term]
                self._idf.pop(term, None)
        self._save()
