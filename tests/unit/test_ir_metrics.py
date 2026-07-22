"""Unit tests for IRMetricsEvaluator — recall/precision/mrr/map/ndcg@k."""

from __future__ import annotations

import math

import pytest

from src.libs.evaluator.base_evaluator import EvalInput
from src.libs.evaluator.ir_metrics_evaluator import IRMetricsEvaluator


def _ndcg_manual(order: list[str], grades: dict[str, int], k: int) -> float:
    """Reference nDCG@k computed by hand for cross-checking the implementation."""
    dcg = 0.0
    for idx, rid in enumerate(order[:k], start=1):
        g = grades.get(rid, 0)
        dcg += (2**g - 1) / math.log2(idx + 1)
    ideal = sorted(grades.values(), reverse=True)[:k]
    idcg = sum((2**g - 1) / math.log2(i + 1) for i, g in enumerate(ideal, start=1))
    return dcg / idcg if idcg > 0 else 0.0


class TestNdcgGraded:
    """graded nDCG must match a hand-computed reference."""

    def test_ndcg_graded_manual(self):
        # retrieved order: d2(g=1), d1(g=3), d3(g=0), d4(g=2)
        grades = {"d1": 3, "d2": 1, "d4": 2}
        retrieved = ["d2", "d1", "d3", "d4"]
        ev = IRMetricsEvaluator(ks=(1, 3, 5))
        result = ev.evaluate([EvalInput("q", retrieved, list(grades), relevance=grades)])

        for k in (1, 3, 5):
            expected = _ndcg_manual(retrieved, grades, k)
            assert result.metrics[f"ndcg@{k}"] == pytest.approx(expected, abs=1e-9)

    def test_ndcg_perfect_ranking_is_one(self):
        grades = {"a": 3, "b": 2, "c": 1}
        retrieved = ["a", "b", "c"]
        ev = IRMetricsEvaluator(ks=(3,))
        result = ev.evaluate([EvalInput("q", retrieved, list(grades), relevance=grades)])
        assert result.metrics["ndcg@3"] == pytest.approx(1.0, abs=1e-9)


class TestRecallPrecision:
    def test_recall_precision_at_k(self):
        # 3 relevant golden, retrieved top-5 hits 2 of them at ranks 1 and 3.
        grades = {"g1": 1, "g2": 1, "g3": 1}
        retrieved = ["g1", "x", "g2", "y", "z"]
        ev = IRMetricsEvaluator(ks=(1, 3, 5))
        result = ev.evaluate([EvalInput("q", retrieved, list(grades), relevance=grades)])

        # capped recall (spec §3.1.4): denom = min(k, |golden|)
        # recall@1: 1/min(1,3)=1.0 ; recall@3: 2/min(3,3)=2/3 ; recall@5: 2/min(5,3)=2/3
        assert result.metrics["recall@1"] == pytest.approx(1.0)
        assert result.metrics["recall@3"] == pytest.approx(2 / 3)
        assert result.metrics["recall@5"] == pytest.approx(2 / 3)
        # precision@1: 1/1 ; precision@3: 2/3 ; precision@5: 2/5
        assert result.metrics["precision@1"] == pytest.approx(1.0)
        assert result.metrics["precision@3"] == pytest.approx(2 / 3)
        assert result.metrics["precision@5"] == pytest.approx(2 / 5)

    def test_k_larger_than_n(self):
        # k=10 but only 2 retrieved -> precision denom is min(k,n)=2.
        grades = {"g1": 1}
        retrieved = ["g1", "x"]
        ev = IRMetricsEvaluator(ks=(10,))
        result = ev.evaluate([EvalInput("q", retrieved, list(grades), relevance=grades)])
        assert result.metrics["precision@10"] == pytest.approx(1 / 2)
        assert result.metrics["recall@10"] == pytest.approx(1.0)

    def test_duplicate_ids_never_exceed_one(self):
        # Document-level fallback: same source repeats across chunks. Metrics
        # must stay <= 1 (regression guard for the >1 bug).
        grades = {"docA": 1}
        retrieved = ["docA", "docA", "docA", "docB", "docC"]  # docA repeated
        ev = IRMetricsEvaluator(ks=(1, 3, 5, 10))
        result = ev.evaluate([EvalInput("q", retrieved, list(grades), relevance=grades)])
        for key, value in result.metrics.items():
            assert 0.0 <= value <= 1.0, f"{key}={value} out of [0,1]"
        assert result.metrics["recall@5"] == pytest.approx(1.0)  # docA found once
        assert result.metrics["precision@5"] == pytest.approx(1 / 3)  # 1 hit / 3 unique


class TestCappedRecall:
    """Capped recall口径 (spec §3.1.4): denominator is min(k, |golden|)."""

    def test_large_golden_capped_to_k(self):
        # 30 relevant golden, k=10, top-10 all hit distinct relevant items.
        grades = {f"g{i}": 1 for i in range(30)}
        retrieved = [f"g{i}" for i in range(10)]
        ev = IRMetricsEvaluator(ks=(10,))
        result = ev.evaluate([EvalInput("q", retrieved, list(grades), relevance=grades)])
        # raw recall would be 10/30 ≈ 0.33 (the假性低分); capped = 10/min(10,30) = 1.0.
        assert result.metrics["recall@10"] == pytest.approx(1.0)

    def test_partial_hits_with_large_golden(self):
        # 30 golden, only 5 of top-10 are relevant -> 5 / min(10,30) = 0.5.
        grades = {f"g{i}": 1 for i in range(30)}
        retrieved = [f"g{i}" for i in range(5)] + ["x1", "x2", "x3", "x4", "x5"]
        ev = IRMetricsEvaluator(ks=(10,))
        result = ev.evaluate([EvalInput("q", retrieved, list(grades), relevance=grades)])
        assert result.metrics["recall@10"] == pytest.approx(0.5)


class TestMapMrr:
    def test_map_at_k(self):
        # relevant at ranks 1 and 3; AP@5 = (1/1 + 2/3) / 2.
        grades = {"g1": 1, "g2": 1}
        retrieved = ["g1", "x", "g2", "y", "z"]
        ev = IRMetricsEvaluator(ks=(5,))
        result = ev.evaluate([EvalInput("q", retrieved, list(grades), relevance=grades)])
        assert result.metrics["map@5"] == pytest.approx((1.0 + 2 / 3) / 2)

    def test_mrr_at_k(self):
        grades = {"g1": 1}
        # first hit at rank 3
        r_in = IRMetricsEvaluator(ks=(3, 5)).evaluate(
            [EvalInput("q", ["x", "y", "g1", "z"], ["g1"], relevance=grades)]
        )
        assert r_in.metrics["mrr@5"] == pytest.approx(1 / 3)
        # first hit outside k=3? hit at rank 3 is within k=3 -> still 1/3
        assert r_in.metrics["mrr@3"] == pytest.approx(1 / 3)
        # hit outside cutoff
        r_out = IRMetricsEvaluator(ks=(2,)).evaluate(
            [EvalInput("q", ["x", "y", "g1"], ["g1"], relevance=grades)]
        )
        assert r_out.metrics["mrr@2"] == pytest.approx(0.0)


class TestBoundaries:
    def test_empty_golden(self):
        ev = IRMetricsEvaluator(ks=(1, 5))
        result = ev.evaluate([EvalInput("q", ["a", "b"], [], relevance={})])
        for key, value in result.metrics.items():
            assert value == 0.0, key

    def test_empty_retrieved(self):
        ev = IRMetricsEvaluator(ks=(1, 5))
        result = ev.evaluate([EvalInput("q", [], ["g1"], relevance={"g1": 2})])
        for key, value in result.metrics.items():
            assert value == 0.0, key

    def test_empty_inputs(self):
        ev = IRMetricsEvaluator(ks=(1, 5))
        result = ev.evaluate([])
        assert result.details["total_queries"] == 0
        assert result.metrics["ndcg@5"] == 0.0

    def test_binary_fallback(self):
        # relevance empty -> golden treated as grade=1; must equal explicit grade=1.
        retrieved = ["g1", "x", "g2"]
        golden = ["g1", "g2"]
        implicit = IRMetricsEvaluator(ks=(3,)).evaluate(
            [EvalInput("q", retrieved, golden)]
        )
        explicit = IRMetricsEvaluator(ks=(3,)).evaluate(
            [EvalInput("q", retrieved, golden, relevance={"g1": 1, "g2": 1})]
        )
        assert implicit.metrics["ndcg@3"] == pytest.approx(explicit.metrics["ndcg@3"])
        assert implicit.metrics["recall@3"] == pytest.approx(explicit.metrics["recall@3"])


class TestAggregationAndDetails:
    def test_per_query_details_present(self):
        ev = IRMetricsEvaluator(ks=(1, 3))
        result = ev.evaluate(
            [
                EvalInput("q1", ["g1"], ["g1"], relevance={"g1": 1}),
                EvalInput("q2", ["x", "g2"], ["g2"], relevance={"g2": 1}),
            ]
        )
        assert result.details["total_queries"] == 2
        rows = result.details["per_query"]
        assert len(rows) == 2
        assert rows[0]["query"] == "q1"
        assert "ndcg@3" in rows[0]

    def test_provider_name(self):
        assert IRMetricsEvaluator().provider_name == "ir"

    def test_metrics_are_mean_over_queries(self):
        # q1 recall@1=1.0, q2 recall@1=0.0 -> mean 0.5
        ev = IRMetricsEvaluator(ks=(1,))
        result = ev.evaluate(
            [
                EvalInput("q1", ["g1"], ["g1"], relevance={"g1": 1}),
                EvalInput("q2", ["x"], ["g2"], relevance={"g2": 1}),
            ]
        )
        assert result.metrics["recall@1"] == pytest.approx(0.5)


class TestPytrecCrosscheck:
    def test_pytrec_crosscheck(self):
        pytest.importorskip("pytrec_eval")
        grades = {"d1": 3, "d2": 1, "d3": 2}
        retrieved = ["d1", "d3", "d2", "x"]
        # use_pytrec=True must not raise and must yield identical self-metrics.
        ev = IRMetricsEvaluator(ks=(1, 3, 5), use_pytrec=True)
        result = ev.evaluate([EvalInput("q", retrieved, list(grades), relevance=grades)])
        expected = _ndcg_manual(retrieved, grades, 3)
        assert result.metrics["ndcg@3"] == pytest.approx(expected, abs=1e-9)
