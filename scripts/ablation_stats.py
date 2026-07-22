#!/usr/bin/env python3
"""ablation_stats.py — statistical significance for ablation comparisons.

Reads the per-query metric breakdown from ``evaluate.py --json`` reports (one per
pipeline configuration) and produces, for each metric:

* a **bootstrap 95% confidence interval** over the per-query metric vector, and
* a **paired permutation test** p-value comparing the baseline (``full``) against
  every other configuration.

This directly answers the "31 samples are not statistically meaningful" critique
by attaching confidence intervals and significance to each ablation delta.

Only ``numpy`` is required (already a transitive dependency); ``scipy`` is not.

Usage:
    python scripts/ablation_stats.py \
        --reports full=experiments/results/full.json \
                  no_rerank=experiments/results/no_rerank.json \
        --metrics ndcg@10 recall@10 \
        --baseline full \
        --out experiments/results/ablation_stats.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

# _DEFAULT_METRICS metrics analysed when --metrics is omitted
_DEFAULT_METRICS = ("ndcg@10", "recall@10")


def _load_report(path: str) -> dict[str, Any]:
    """Load a single evaluate.py --json report."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "per_query" not in data:
        raise ValueError(f"Report {path} has no 'per_query' field")
    return data


def _metric_vector(report: dict[str, Any], metric: str) -> dict[str, float]:
    """Extract a query -> metric value map from a report's per_query rows."""
    out: dict[str, float] = {}
    for row in report.get("per_query", []):
        query = str(row.get("query", ""))
        if metric in row:
            out[query] = float(row[metric])
    return out


def _bootstrap_ci(
    values: np.ndarray, n_samples: int, rng: np.random.Generator
) -> tuple[float, float, float]:
    """Return (mean, ci_low, ci_high) via bootstrap resampling of the mean."""
    if values.size == 0:
        return 0.0, 0.0, 0.0
    idx = rng.integers(0, values.size, size=(n_samples, values.size))
    means = values[idx].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(values.mean()), float(low), float(high)


def _paired_permutation_p(
    baseline: np.ndarray, other: np.ndarray, n_perm: int, rng: np.random.Generator
) -> float:
    """Two-sided paired permutation test via random sign flips of the diffs."""
    if baseline.size == 0 or baseline.size != other.size:
        return float("nan")
    diff = baseline - other
    observed = float(np.abs(diff.mean()))
    if observed == 0.0:
        return 1.0
    signs = rng.choice([-1.0, 1.0], size=(n_perm, diff.size))
    permuted = np.abs((signs * diff).mean(axis=1))
    count = int(np.sum(permuted >= observed - 1e-12))
    return (count + 1) / (n_perm + 1)


def _aligned_vectors(
    base_map: dict[str, float], other_map: dict[str, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Align two query->value maps into paired arrays over shared queries."""
    shared = [q for q in base_map if q in other_map]
    base = np.array([base_map[q] for q in shared], dtype=float)
    other = np.array([other_map[q] for q in shared], dtype=float)
    return base, other


def compute_stats(
    reports: dict[str, dict[str, Any]],
    metrics: list[str],
    baseline_label: str,
    bootstrap_samples: int,
    n_perm: int,
    seed: int = 12345,
) -> dict[str, Any]:
    """Compute CI + significance for every (metric, config) pair."""
    rng = np.random.default_rng(seed)
    if baseline_label not in reports:
        raise ValueError(f"Baseline '{baseline_label}' not among reports: {list(reports)}")

    out: dict[str, Any] = {
        "baseline": baseline_label,
        "bootstrap_samples": bootstrap_samples,
        "permutations": n_perm,
        "metrics": {},
    }

    for metric in metrics:
        base_map = _metric_vector(reports[baseline_label], metric)
        per_config: dict[str, Any] = {}
        for label, report in reports.items():
            values_map = _metric_vector(report, metric)
            values = np.array(list(values_map.values()), dtype=float)
            mean, low, high = _bootstrap_ci(values, bootstrap_samples, rng)
            entry: dict[str, Any] = {
                "mean": round(mean, 4),
                "ci95": [round(low, 4), round(high, 4)],
                "n": int(values.size),
            }
            if label != baseline_label:
                b_vec, o_vec = _aligned_vectors(base_map, values_map)
                p = _paired_permutation_p(b_vec, o_vec, n_perm, rng)
                entry["p_vs_baseline"] = None if np.isnan(p) else round(p, 4)
                entry["significant"] = bool(not np.isnan(p) and p < 0.05)
            per_config[label] = entry
        out["metrics"][metric] = per_config

    return out


def _print_table(stats: dict[str, Any]) -> None:
    """Render a human-readable comparison table to stderr."""
    baseline = stats["baseline"]
    for metric, per_config in stats["metrics"].items():
        print(f"\n=== {metric} (baseline={baseline}) ===", file=sys.stderr)
        header = f"{'Pipeline':<16} {'mean':>7} {'95% CI':>20} {'p(vs base)':>12}"
        print(header, file=sys.stderr)
        print("-" * len(header), file=sys.stderr)
        for label, entry in per_config.items():
            ci = entry["ci95"]
            ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]"
            if label == baseline:
                p_str = "-"
            else:
                p = entry.get("p_vs_baseline")
                mark = " *" if entry.get("significant") else ""
                p_str = "n/a" if p is None else f"{p:.4f}{mark}"
            print(
                f"{label:<16} {entry['mean']:>7.3f} {ci_str:>20} {p_str:>12}",
                file=sys.stderr,
            )


def _parse_reports(pairs: list[str]) -> dict[str, dict[str, Any]]:
    """Parse ``label=path`` pairs into an ordered mapping of loaded reports."""
    reports: dict[str, dict[str, Any]] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"--reports item must be label=path, got: {pair!r}")
        label, path = pair.split("=", 1)
        reports[label.strip()] = _load_report(path.strip())
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablation statistical significance")
    parser.add_argument(
        "--reports",
        nargs="+",
        required=True,
        help="One or more label=path pairs of evaluate.py --json reports",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(_DEFAULT_METRICS),
        help=f"Per-query metric keys to analyse (default: {' '.join(_DEFAULT_METRICS)})",
    )
    parser.add_argument("--baseline", default="full", help="Baseline config label")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument(
        "--out",
        default="experiments/results/ablation_stats.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    try:
        reports = _parse_reports(args.reports)
        stats = compute_stats(
            reports,
            args.metrics,
            args.baseline,
            args.bootstrap_samples,
            args.permutations,
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"\n❌ {exc}", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    _print_table(stats)
    print(f"\nSaved stats to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
