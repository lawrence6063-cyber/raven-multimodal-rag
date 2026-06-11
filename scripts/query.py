#!/usr/bin/env python3
"""Query script — CLI entry point for searching the knowledge base.

Usage:
    python scripts/query.py --query "your question" [--top-k 10] [--collection x] [--verbose] [--no-rerank]
    python scripts/query.py --image data/query_images/figure.png        # 以图搜文/图
    python scripts/query.py --query "what is this?" --image data/q.png   # 图文混合
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.settings import load_settings, SettingsError
from src.core.query_engine.hybrid_search import HybridSearch
from src.core.query_engine.reranker import QueryReranker
from src.observability.logger import get_logger

logger = get_logger("query")


def main():
    parser = argparse.ArgumentParser(description="Query the RAG knowledge base")
    parser.add_argument("--query", default="", help="Search query text")
    parser.add_argument("--image", default=None, help="Query image path (cross-modal search)")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results (default: 10)")
    parser.add_argument("--collection", default=None, help="Limit to specific collection")
    parser.add_argument("--verbose", action="store_true", help="Show detailed intermediate results")
    parser.add_argument("--no-rerank", action="store_true", help="Skip reranking step")
    args = parser.parse_args()

    if not args.query and not args.image:
        parser.error("provide at least one of --query or --image")

    # Load settings
    try:
        settings = load_settings()
    except SettingsError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)

    # Build filters
    filters = {}
    if args.collection:
        filters["collection"] = args.collection

    # Search
    try:
        hybrid = HybridSearch(settings)
        results = hybrid.search(
            query=args.query, top_k=args.top_k, filters=filters or None, image=args.image
        )
    except Exception as e:
        logger.error(f"Search failed: {e}")
        print(f"\n❌ Search failed: {e}")
        print("💡 Hint: Have you ingested documents? Run: python scripts/ingest.py --path <folder>")
        sys.exit(1)

    # Rerank (text-driven; skip for image-only queries)
    if args.query and not args.no_rerank and settings.rerank.enabled:
        try:
            reranker = QueryReranker(settings)
            results = reranker.rerank(args.query, results)
        except Exception as e:
            logger.warning(f"Reranking failed, using fusion results: {e}")

    label = args.query or f"[image: {args.image}]"

    # Output results
    if not results:
        print(f"\n🔍 No results found for: \"{label}\"")
        print("💡 Hint: Have you ingested documents? Run: python scripts/ingest.py --path <folder>")
        return

    print(f"\n🔍 Query: \"{label}\"")
    print(f"📊 Results: {len(results)} (top-{args.top_k})")
    print("=" * 60)

    for i, r in enumerate(results, 1):
        source = r.metadata.get("source_path", r.metadata.get("file_name", "unknown"))
        print(f"\n[{i}] Score: {r.score:.4f}")
        print(f"    Source: {source}")
        if r.metadata.get("chunk_index") is not None:
            print(f"    Chunk: #{r.metadata['chunk_index']}")
        # Show text preview
        preview = r.text[:200].replace('\n', ' ') if r.text else "(no text)"
        print(f"    Text: {preview}...")

        if args.verbose:
            print(f"    ID: {r.chunk_id}")
            print(f"    Metadata: {r.metadata}")

    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    main()
