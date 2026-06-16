#!/usr/bin/env python3
"""Rebuild BM25 index from all text chunks in Chroma."""

import os
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.settings import load_settings
from src.core.types import ChunkRecord
from src.ingestion.storage.bm25_indexer import BM25Indexer

import chromadb
from chromadb.config import Settings as CS


def tokenize(text: str) -> dict[str, float]:
    """Simple word tokenization for BM25 sparse vector."""
    words = re.findall(r'[a-zA-Z]+', text.lower())
    return dict(Counter(words))


def main():
    s = load_settings()
    client = chromadb.PersistentClient(
        path=s.vector_store.persist_directory,
        settings=CS(anonymized_telemetry=False),
    )
    col = client.get_or_create_collection('default')

    print("Reading all text chunks from Chroma...")
    results = col.get(limit=10000, include=['documents', 'metadatas'])

    text_chunks = []
    for i in range(len(results['ids'])):
        meta = results['metadatas'][i] or {}
        doc = results['documents'][i] or ''
        if meta.get('modality') == 'image':
            continue
        if not doc.strip():
            continue
        text_chunks.append((results['ids'][i], doc))

    print(f"Found {len(text_chunks)} text chunks")

    records = []
    for id_, doc in text_chunks:
        records.append(ChunkRecord(
            id=id_,
            text=doc,
            dense_vector=[],
            sparse_vector=tokenize(doc),
            metadata={},
        ))

    # Remove old index
    index_file = os.path.join(s.ingestion.bm25_index_path, 'bm25_index.pkl')
    if os.path.exists(index_file):
        os.remove(index_file)
        print("Removed old BM25 index")

    # Rebuild
    bm25 = BM25Indexer(s.ingestion.bm25_index_path)
    bm25.build(records)
    print(f"✅ BM25 index rebuilt: doc_count={bm25._doc_count}, "
          f"avg_doc_length={bm25._avg_doc_length:.1f}, "
          f"vocabulary={len(bm25._idf)} terms")

    # Verify
    test = bm25.query(['transformer', 'attention'], top_k=3)
    print(f"\nVerification query ['transformer', 'attention']: {len(test)} results")
    for r in test:
        print(f"  {r['chunk_id'][:30]}... score={r['score']:.3f}")


if __name__ == "__main__":
    main()
