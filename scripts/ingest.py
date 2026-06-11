#!/usr/bin/env python3
"""Ingestion script — CLI entry point for document ingestion.

Usage:
    python scripts/ingest.py --path <file_or_folder> [--collection name] [--force]
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.settings import load_settings, SettingsError
from src.ingestion.pipeline import IngestionPipeline
from src.observability.logger import get_logger

logger = get_logger("ingest")


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG knowledge base")
    parser.add_argument("--path", required=True, help="Path to PDF file or folder of PDFs")
    parser.add_argument("--collection", default="default", help="Collection name (default: 'default')")
    parser.add_argument("--force", action="store_true", help="Force re-processing even if already ingested")
    args = parser.parse_args()

    # Load settings
    try:
        settings = load_settings()
    except SettingsError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)

    # Discover files
    target = Path(args.path)
    if target.is_file():
        files = [target]
    elif target.is_dir():
        files = sorted(target.glob("**/*.pdf"))
    else:
        logger.error(f"Path not found: {args.path}")
        sys.exit(1)

    if not files:
        logger.warning(f"No PDF files found in: {args.path}")
        sys.exit(0)

    # Run pipeline
    pipeline = IngestionPipeline(settings)
    success_count = 0
    skip_count = 0
    fail_count = 0

    for i, file in enumerate(files, 1):
        logger.info(f"[{i}/{len(files)}] Processing: {file.name}")
        try:
            result = pipeline.run(str(file), collection=args.collection, force=args.force)
            if result["status"] == "skipped":
                skip_count += 1
                logger.info(f"  → Skipped (already processed)")
            else:
                success_count += 1
                logger.info(f"  → Success: {result['chunk_count']} chunks")
        except Exception as e:
            fail_count += 1
            logger.error(f"  → Failed: {e}")

    # Summary
    logger.info(f"\n{'='*50}")
    logger.info(f"Ingestion complete: {success_count} success, {skip_count} skipped, {fail_count} failed")
    logger.info(f"Collection: {args.collection}")


if __name__ == "__main__":
    main()
