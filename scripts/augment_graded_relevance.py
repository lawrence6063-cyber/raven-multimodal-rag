#!/usr/bin/env python3
"""augment_graded_relevance.py — widen chunk-level golden to graded relevance.

For each test case that pins a single correct chunk (``expected_chunk_ids`` /
``relevance`` grade 3), this marks the same-document neighboring chunks (within
``--window`` positions by ``chunk_index``) as grade 1 (partially relevant),
following BEIR-style multi-level qrels. This makes retrieval of an adjacent chunk
from the correct document count as a partial hit instead of a total miss, which
better reflects real retrieval quality than single-chunk exact match.

Neighbors are resolved against the live vector store so their chunk_ids are real.
Only reads the store (no LLM / embedding key required).

Usage:
    python scripts/augment_graded_relevance.py data/golden_synth_li.json \
        --out data/golden_synth_li.json --window 2 --neighbor-grade 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.settings import load_settings
from src.libs.vector_store.vector_store_factory import VectorStoreFactory


def _parse_chunk_id(chunk_id: str) -> tuple[str, int] | None:
    """Parse ``doc_<hash>_<index>_<contenthash>`` into (doc_key, index).

    Returns None for ids that do not follow the document-chunk layout (e.g.
    image chunks ``img_...``), which have no positional neighbors.
    """
    parts = chunk_id.split("_")
    if len(parts) < 4 or parts[0] != "doc":
        return None
    try:
        index = int(parts[2])
    except ValueError:
        return None
    doc_key = "_".join(parts[:2])  # doc_<hash>
    return doc_key, index


def _build_doc_index(store) -> dict[str, dict[int, str]]:
    """Map doc_key -> {chunk_index -> chunk_id} from the vector store."""
    doc_index: dict[str, dict[int, str]] = {}
    for rec in store.get_all():
        parsed = _parse_chunk_id(rec.id)
        if parsed is None:
            continue
        doc_key, index = parsed
        doc_index.setdefault(doc_key, {})[index] = rec.id
    return doc_index


def augment(
    cases: list[dict[str, Any]],
    doc_index: dict[str, dict[int, str]],
    window: int,
    correct_grade: int,
    neighbor_grade: int,
) -> tuple[int, int]:
    """Augment cases in place. Returns (num_augmented, num_neighbors_added)."""
    augmented = 0
    neighbors_added = 0
    for case in cases:
        chunk_ids = case.get("expected_chunk_ids") or []
        if not chunk_ids:
            continue
        relevance: dict[str, int] = {}
        touched = False
        for cid in chunk_ids:
            relevance[cid] = max(relevance.get(cid, 0), correct_grade)
            parsed = _parse_chunk_id(cid)
            if parsed is None:
                continue
            doc_key, index = parsed
            neighbors = doc_index.get(doc_key, {})
            for offset in range(-window, window + 1):
                if offset == 0:
                    continue
                nb = neighbors.get(index + offset)
                if nb and nb not in relevance:
                    relevance[nb] = neighbor_grade
                    neighbors_added += 1
                    touched = True
        case["relevance"] = relevance
        if touched:
            augmented += 1
    return augmented, neighbors_added


def main() -> None:
    parser = argparse.ArgumentParser(description="Widen golden set to graded relevance")
    parser.add_argument("input", help="Input golden JSON file")
    parser.add_argument("--out", required=True, help="Output JSON path")
    parser.add_argument("--window", type=int, default=2, help="Neighbor window by chunk_index")
    parser.add_argument("--correct-grade", type=int, default=3)
    parser.add_argument("--neighbor-grade", type=int, default=1)
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    cases = data.get("test_cases")
    if not isinstance(cases, list):
        print("\n❌ input has no 'test_cases' list", file=sys.stderr)
        sys.exit(1)

    settings = load_settings()
    store = VectorStoreFactory.create(settings.vector_store)
    doc_index = _build_doc_index(store)

    augmented, neighbors_added = augment(
        cases, doc_index, args.window, args.correct_grade, args.neighbor_grade
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"augmented {augmented} cases, added {neighbors_added} neighbor grades -> {out_path}")


if __name__ == "__main__":
    main()
