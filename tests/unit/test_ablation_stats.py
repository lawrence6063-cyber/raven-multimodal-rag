"""Unit tests for scripts/ablation_stats.py — bootstrap CI + permutation test."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "ablation_stats.py"
_spec = importlib.util.spec_from_file_location("ablation_stats", _SCRIPT)
ablation_stats = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ablation_stats)  # type: ignore[union-attr]


def _report(values: list[float], metric: str = "ndcg@10") -> dict:
    return {"per_query": [{"query": f"q{i}", metric: v} for i, v in enumerate(values)]}


class TestBootstrapCI:
    def test_ci_contains_mean_and_ordered(self):
        rng = np.random.default_rng(0)
        values = np.array([0.9, 0.8, 0.85, 0.95, 0.7, 0.88, 0.92, 0.6, 0.75, 0.83])
        mean, low, high = ablation_stats._bootstrap_ci(values, 1000, rng)
        assert low <= mean <= high
        assert mean == pytest.approx(values.mean())

    def test_empty_vector(self):
        rng = np.random.default_rng(0)
        assert ablation_stats._bootstrap_ci(np.array([]), 100, rng) == (0.0, 0.0, 0.0)


class TestPermutation:
    def test_significant_difference(self):
        rng = np.random.default_rng(1)
        base = np.array([0.9, 0.8, 0.85, 0.95, 0.7, 0.88, 0.92, 0.6, 0.75, 0.83])
        other = np.array([0.6, 0.5, 0.55, 0.65, 0.4, 0.58, 0.62, 0.3, 0.45, 0.53])
        p = ablation_stats._paired_permutation_p(base, other, 10000, rng)
        assert p < 0.05

    def test_identical_vectors_p_one(self):
        rng = np.random.default_rng(1)
        v = np.array([0.5, 0.6, 0.7])
        p = ablation_stats._paired_permutation_p(v, v.copy(), 5000, rng)
        assert p == pytest.approx(1.0)

    def test_mismatched_size_returns_nan(self):
        rng = np.random.default_rng(1)
        p = ablation_stats._paired_permutation_p(
            np.array([0.5, 0.6]), np.array([0.5]), 100, rng
        )
        assert np.isnan(p)


class TestComputeStats:
    def test_end_to_end(self):
        reports = {
            "full": _report([0.9, 0.8, 0.85, 0.95, 0.7, 0.88, 0.92, 0.6, 0.75, 0.83]),
            "no_rerank": _report([0.6, 0.5, 0.55, 0.65, 0.4, 0.58, 0.62, 0.3, 0.45, 0.53]),
        }
        stats = ablation_stats.compute_stats(reports, ["ndcg@10"], "full", 500, 5000)
        assert stats["baseline"] == "full"
        full = stats["metrics"]["ndcg@10"]["full"]
        no_rerank = stats["metrics"]["ndcg@10"]["no_rerank"]
        assert full["n"] == 10
        assert "p_vs_baseline" not in full  # baseline has no p
        assert no_rerank["p_vs_baseline"] is not None
        assert no_rerank["significant"] is True

    def test_unknown_baseline_raises(self):
        reports = {"full": _report([0.5])}
        with pytest.raises(ValueError):
            ablation_stats.compute_stats(reports, ["ndcg@10"], "missing", 100, 100)

    def test_parse_reports_requires_equals(self):
        with pytest.raises(ValueError):
            ablation_stats._parse_reports(["no_equals_sign"])
