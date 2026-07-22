#!/usr/bin/env python3
"""reembed_index.py — re-embed stored text chunks with the CURRENT embedding.

Diagnosis (2026-07-16) showed the persisted chunk vectors are in a different
embedding space than the current query-time embedding: re-embedding a stored
chunk's own text yields cosine ~0.16 against its stored vector (should be ~1.0).
Dense retrieval therefore returns the same chunks for every query.

This script fixes the misalignment WITHOUT re-parsing PDFs: it reads each
chunk's stored ``document`` text from Chroma, re-encodes it with the configured
embedding provider, and updates the vector in place. Image-modality vectors are
left untouched (they belong to the cross-modal path and are keyed by image, not
text). The run is resumable via a checkpoint file.

Usage:
    export DASHSCOPE_API_KEY=...
    python scripts/reembed_index.py [--workers 6] [--batch-size 64]
        [--collection default] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.settings import load_settings
from src.libs.embedding.embedding_factory import EmbeddingFactory
from src.observability.logger import get_logger

logger = get_logger("reembed_index")

# _CHECKPOINT default path storing already re-embedded chunk ids (one per line)
_CHECKPOINT = "experiments/results/reembed_checkpoint.txt"


def _embed_one(embedding, text: str, retries: int = 3) -> list[float] | None:
    """Embed a single text with retries, returning None on persistent failure."""
    for attempt in range(1, retries + 1):
        try:
            vectors = embedding.embed([text])
            if vectors and vectors[0]:
                return vectors[0]
            return None
        except Exception as exc:  # noqa: BLE001 - retry transient API errors
            if attempt == retries:
                logger.warning(f"embed failed after {retries} tries: {exc}")
                return None
            time.sleep(1.5 * attempt)
    return None


def _load_checkpoint(path: Path) -> set[str]:
    """Load the set of already-processed chunk ids from the checkpoint file."""
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-embed stored text chunks in place")
    parser.add_argument("--workers", type=int, default=6, help="Concurrent embedding calls")
    parser.add_argument("--batch-size", type=int, default=64, help="Chunks per Chroma update")
    parser.add_argument("--collection", default=None, help="Override collection name")
    parser.add_argument("--limit", type=int, default=0, help="Process at most N chunks (0=all)")
    parser.add_argument("--checkpoint", default=_CHECKPOINT)
    args = parser.parse_args()

    import chromadb

    settings = load_settings()
    collection_name = args.collection or settings.vector_store.collection_name
    embedding = EmbeddingFactory.create(settings.embedding)

    client = chromadb.PersistentClient(path=settings.vector_store.persist_directory)
    col = client.get_collection(collection_name)

    got = col.get(include=["documents", "metadatas"])
    ids = got["ids"]
    docs = got["documents"]
    metas = got["metadatas"]

    # Only text chunks with a stored document; skip image-modality vectors.
    targets = [
        (cid, doc)
        for cid, doc, meta in zip(ids, docs, metas)
        if doc and (meta or {}).get("modality") != "image"
    ]

    ckpt_path = Path(args.checkpoint)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    done = _load_checkpoint(ckpt_path)
    remaining = [(cid, doc) for cid, doc in targets if cid not in done]
    if args.limit > 0:
        remaining = remaining[: args.limit]

    total = len(remaining)
    logger.info(
        f"collection={collection_name} total_chunks={len(ids)} text_targets={len(targets)} "
        f"already_done={len(done)} to_process={total} workers={args.workers}"
    )
    if total == 0:
        logger.info("Nothing to do — index already re-embedded.")
        return

    processed = 0
    failed = 0
    start = time.perf_counter()

    with ckpt_path.open("a", encoding="utf-8") as ckpt:
        for base in range(0, total, args.batch_size):
            batch = remaining[base : base + args.batch_size]

            # Serial embedding: the multimodal DashScope client is not
            # thread-safe (concurrency hangs), and a single call is ~0.3s.
            up_ids: list[str] = []
            up_vecs: list[list[float]] = []
            for cid, doc in batch:
                vec = _embed_one(embedding, doc)
                if vec is None:
                    failed += 1
                    continue
                up_ids.append(cid)
                up_vecs.append(vec)

            if up_ids:
                col.update(ids=up_ids, embeddings=up_vecs)
                ckpt.write("\n".join(up_ids) + "\n")
                ckpt.flush()

            processed += len(up_ids)
            elapsed = time.perf_counter() - start
            rate = processed / elapsed if elapsed > 0 else 0.0
            eta = (total - processed - failed) / rate if rate > 0 else 0.0
            logger.info(
                f"progress {processed + failed}/{total} (ok={processed} fail={failed}) "
                f"rate={rate:.1f}/s eta={eta / 60:.1f}min"
            )

    logger.info(
        f"DONE re-embedded ok={processed} failed={failed} "
        f"elapsed={(time.perf_counter() - start) / 60:.1f}min"
    )
    # Emit a machine-readable summary line.
    print(json.dumps({"ok": processed, "failed": failed, "total": total}))


if __name__ == "__main__":
    main()
