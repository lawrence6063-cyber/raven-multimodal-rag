#!/usr/bin/env python3
"""Quick test: run 1 query to verify setup before full ablation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.settings import load_settings
from src.core.query_engine.hybrid_search import HybridSearch
from src.data_pipeline.vector_store.chroma_store import ChromaStore

def main():
    print("=== Quick Test: Setup Verification ===\n")

    settings = load_settings()
    print(f"✓ Settings loaded")
    print(f"  LLM: {settings.llm.provider}/{settings.llm.model}")
    print(f"  Embedding: {settings.embedding.provider}/{settings.embedding.model}")
    print(f"  Golden set: {settings.evaluation.golden_test_set}")

    # Check vector store
    store = ChromaStore(settings)
    count = store.count()
    print(f"\n✓ ChromaDB connected: {count} chunks indexed")

    if count == 0:
        print("\n❌ Vector store is empty! Run 'python3 scripts/ingest.py --path data/documents' first.")
        sys.exit(1)

    # Sample retrieval
    hybrid = HybridSearch(settings)
    print("\n✓ Running test query: 'What is RAG?'")
    results = hybrid.search("What is Retrieval-Augmented Generation?", top_k=3)

    print(f"  Retrieved {len(results)} results:")
    for i, r in enumerate(results[:3], 1):
        src = r.metadata.get('source_path', 'unknown')[:60]
        print(f"    {i}. {src} (score: {r.score:.3f})")

    print("\n✅ Setup verified! Ready to run full ablation with 'bash scripts/run_ablation.sh'")

if __name__ == "__main__":
    main()
