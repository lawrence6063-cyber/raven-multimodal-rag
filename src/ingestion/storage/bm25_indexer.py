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
        """Incrementally add records to BM25 index (merge, not overwrite).

        Loads existing index from disk, merges new records, recomputes IDF,
        then persists the combined index.
        """
        if not records:
            return

        # Load existing index first (incremental merge)
        if not self._loaded:
            self._load()

        # Deduplicate: skip records whose IDs already exist in the index
        existing_ids: set[str] = set()
        for postings in self._inverted_index.values():
            for p in postings:
                existing_ids.add(p["chunk_id"])

        new_records = [r for r in records if r.id not in existing_ids]
        if not new_records:
            return

        # Merge new records into existing inverted index
        inverted: dict[str, list[dict]] = defaultdict(list, {k: list(v) for k, v in self._inverted_index.items()})
        new_doc_lengths = []

        for rec in new_records:
            doc_len = sum(rec.sparse_vector.values()) if rec.sparse_vector else 0
            new_doc_lengths.append(doc_len)
            for term, tf in rec.sparse_vector.items():
                inverted[term].append({"chunk_id": rec.id, "tf": tf, "doc_length": doc_len})

        self._inverted_index = dict(inverted)

        # Recompute doc_count and avg_doc_length incrementally
        old_total_length = self._avg_doc_length * self._doc_count
        self._doc_count += len(new_records)
        new_total_length = old_total_length + sum(new_doc_lengths)
        self._avg_doc_length = new_total_length / self._doc_count if self._doc_count > 0 else 0

        # Recompute IDF for all terms
        self._idf = {}
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
