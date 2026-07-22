#!/usr/bin/env python3
"""merge_testset.py — merge multiple golden test-set files with dedup.

Combines several ``{"test_cases": [...]}`` files into one, de-duplicating by a
normalized query key (strip + lower + whitespace folding). On conflict the entry
carrying more information wins (``expected_chunk_ids`` > ``relevance`` >
``answer`` presence), so hand-written chunk-level cases are never overwritten by
weaker synthetic ones.

Usage:
    python scripts/merge_testset.py \
        data/golden_papers.json data/golden_synth_ragas.json data/golden_synth_li.json \
        --out data/golden_papers.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

_WS_RE = re.compile(r"\s+")


def _normalize_query(query: str) -> str:
    """Normalize a query for dedup: strip, lower, fold internal whitespace."""
    return _WS_RE.sub(" ", query.strip().lower())


def _richness(case: dict[str, Any]) -> int:
    """Score how much information a case carries (higher wins on conflict)."""
    score = 0
    if case.get("expected_chunk_ids"):
        score += 4
    if case.get("relevance"):
        score += 2
    if case.get("answer"):
        score += 1
    if case.get("expected_sources"):
        score += 1
    return score


def _load_cases(path: str) -> list[dict[str, Any]]:
    """Load the ``test_cases`` list from a golden file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = data.get("test_cases")
    if not isinstance(cases, list):
        raise ValueError(f"{path}: missing 'test_cases' list")
    return cases


def merge_cases(files: list[str]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Merge and dedup cases from multiple files.

    Returns:
        (merged_cases, stats) where stats counts entries per ``source``.
    """
    by_query: dict[str, dict[str, Any]] = {}
    order: list[str] = []

    for path in files:
        for case in _load_cases(path):
            query = str(case.get("query", "")).strip()
            if not query:
                continue
            key = _normalize_query(query)
            if key not in by_query:
                by_query[key] = case
                order.append(key)
            elif _richness(case) > _richness(by_query[key]):
                by_query[key] = case

    merged = [by_query[k] for k in order]

    stats: dict[str, int] = {}
    for case in merged:
        src = str(case.get("source", "unknown"))
        stats[src] = stats.get(src, 0) + 1
    return merged, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge golden test-set files")
    parser.add_argument("files", nargs="+", help="Input golden JSON files")
    parser.add_argument("--out", required=True, help="Output JSON path")
    parser.add_argument(
        "--description",
        default="Merged golden test set (dedup by normalized query).",
        help="Top-level description string for the output file",
    )
    args = parser.parse_args()

    try:
        merged, stats = merge_cases(args.files)
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"\n❌ {exc}", file=sys.stderr)
        sys.exit(1)

    n_chunk_level = sum(1 for c in merged if c.get("expected_chunk_ids"))
    out = {
        "description": args.description,
        "test_cases": merged,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Merged {len(merged)} unique cases -> {out_path}", file=sys.stderr)
    print(f"  chunk-level (expected_chunk_ids): {n_chunk_level}", file=sys.stderr)
    print("  by source:", file=sys.stderr)
    for src, count in sorted(stats.items()):
        print(f"    {src:<20} {count}", file=sys.stderr)


if __name__ == "__main__":
    main()
