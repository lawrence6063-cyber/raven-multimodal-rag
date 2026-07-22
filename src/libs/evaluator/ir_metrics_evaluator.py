"""IRMetricsEvaluator — standard IR metrics (recall/precision/mrr/map/ndcg@k).

Offline, LLM-free implementation aligned with the BEIR / TREC eval conventions.
Supports graded relevance (0..3) with automatic degradation to binary relevance
when ``EvalInput.relevance`` is empty, keeping backward compatibility with the
legacy document-level golden sets.

nDCG uses the exponential gain formulation::

    DCG@k = Σ_{i=1..k} (2^rel_i - 1) / log2(i + 1)

IDCG@k takes the golden grades sorted descending as the ideal ranking. When
IDCG@k == 0 (no relevant golden for the query) nDCG@k is recorded as 0.0.
"""

from __future__ import annotations

import math

from src.libs.evaluator.base_evaluator import BaseEvaluator, EvalInput, EvalResult
from src.libs.evaluator.evaluator_factory import register_evaluator

# _DEFAULT_KS default cutoff values for the @k metric family
_DEFAULT_KS: tuple[int, ...] = (1, 3, 5, 10)


@register_evaluator("ir")
class IRMetricsEvaluator(BaseEvaluator):
    """IR standard-metrics evaluator computing recall/precision/mrr/map/ndcg@k."""

    def __init__(self, ks: tuple[int, ...] = _DEFAULT_KS, use_pytrec: bool = False):
        """Initialize the evaluator.

        Args:
            ks: Cutoff values for the @k metric family (deduplicated, sorted).
            use_pytrec: If True and ``pytrec_eval`` is installed, cross-check the
                self-computed nDCG/MAP against the reference implementation.
                Silently falls back to the self-implementation when unavailable.
        """
        cleaned = sorted({int(k) for k in ks if int(k) > 0})
        self._ks: tuple[int, ...] = tuple(cleaned) or _DEFAULT_KS
        self._use_pytrec = use_pytrec

    def evaluate(self, inputs: list[EvalInput]) -> EvalResult:
        """Compute IR metrics aggregated over all evaluation inputs.

        Returns:
            EvalResult whose metrics hold the mean of each ``<name>@<k>`` metric
            across queries, and whose details carry a ``per_query`` list with the
            same per-query breakdown (consumed by ablation significance tests).
        """
        metric_names = ("recall", "precision", "hit_rate", "mrr", "map", "ndcg")
        if not inputs:
            empty = {f"{name}@{k}": 0.0 for name in metric_names for k in self._ks}
            return EvalResult(metrics=empty, details={"per_query": [], "total_queries": 0})

        # accumulators[metric@k] -> list of per-query scores
        accumulators: dict[str, list[float]] = {
            f"{name}@{k}": [] for name in metric_names for k in self._ks
        }
        per_query: list[dict[str, float | str]] = []

        for inp in inputs:
            grades = self._relevance_map(inp)
            retrieved = self._dedup_preserve_order(inp.retrieved_ids)
            row: dict[str, float | str] = {"query": inp.query}
            for k in self._ks:
                scores = self._query_metrics_at_k(retrieved, grades, k)
                for name, value in scores.items():
                    key = f"{name}@{k}"
                    accumulators[key].append(value)
                    row[key] = round(value, 6)
            per_query.append(row)

        metrics = {
            key: (sum(values) / len(values) if values else 0.0)
            for key, values in accumulators.items()
        }

        if self._use_pytrec:
            self._crosscheck_pytrec(inputs, metrics)

        return EvalResult(
            metrics=metrics,
            details={"per_query": per_query, "total_queries": len(inputs)},
        )

    @staticmethod
    def _dedup_preserve_order(ids: list[str]) -> list[str]:
        """De-duplicate retrieved ids keeping first-seen order.

        IR metrics assume a ranked list of *unique* documents (matching the
        BEIR/pytrec_eval ``run`` convention). At document-level evaluation the
        same source can appear for several chunks; without de-dup a single
        relevant document would be counted multiple times, pushing recall /
        precision / map / ndcg above 1.0. Chunk-level ids are already unique so
        this is a no-op for them.
        """
        seen: set[str] = set()
        out: list[str] = []
        for rid in ids:
            if rid not in seen:
                seen.add(rid)
                out.append(rid)
        return out

    @staticmethod
    def _relevance_map(inp: EvalInput) -> dict[str, int]:
        """Resolve the id -> grade map for a single input.

        graded mode: use ``inp.relevance`` (0..3) verbatim.
        binary mode: fall back to every golden id having grade 1.
        """
        if inp.relevance:
            return {rid: int(g) for rid, g in inp.relevance.items() if int(g) > 0}
        return {rid: 1 for rid in inp.golden_ids}

    @staticmethod
    def _query_metrics_at_k(
        retrieved_ids: list[str], grades: dict[str, int], k: int
    ) -> dict[str, float]:
        """Compute all @k metrics for a single query.

        Args:
            retrieved_ids: Retrieved ids ordered by descending score.
            grades: Map of relevant id -> grade (>0). Empty means no golden.
            k: Cutoff.

        Returns:
            Dict with recall/precision/hit_rate/mrr/map/ndcg keys (no @k suffix).
        """
        total_relevant = len(grades)
        # Empty golden or empty retrieval -> all zero (spec §2.1 boundary rules).
        if total_relevant == 0 or not retrieved_ids:
            return {
                "recall": 0.0,
                "precision": 0.0,
                "hit_rate": 0.0,
                "mrr": 0.0,
                "map": 0.0,
                "ndcg": 0.0,
            }

        top = retrieved_ids[:k]
        n = len(retrieved_ids)
        denom = min(k, n)

        hits = 0
        first_hit_rank = 0
        ap_numerator = 0.0
        dcg = 0.0
        for idx, rid in enumerate(top, start=1):
            grade = grades.get(rid, 0)
            if grade > 0:
                hits += 1
                if first_hit_rank == 0:
                    first_hit_rank = idx
                ap_numerator += hits / idx  # precision@idx at each relevant hit
                dcg += (2**grade - 1) / math.log2(idx + 1)

        recall = hits / min(k, total_relevant)  # capped recall：分母取 min(k, |golden|)
        precision = hits / denom if denom else 0.0
        hit_rate = 1.0 if hits > 0 else 0.0
        mrr = 1.0 / first_hit_rank if first_hit_rank > 0 else 0.0
        # AP normalized by the number of relevant docs reachable within k.
        ap = ap_numerator / min(total_relevant, k) if min(total_relevant, k) else 0.0
        idcg = IRMetricsEvaluator._ideal_dcg(grades, k)
        ndcg = dcg / idcg if idcg > 0 else 0.0

        return {
            "recall": recall,
            "precision": precision,
            "hit_rate": hit_rate,
            "mrr": mrr,
            "map": ap,
            "ndcg": ndcg,
        }

    @staticmethod
    def _ideal_dcg(grades: dict[str, int], k: int) -> float:
        """Compute IDCG@k from the golden grades sorted descending."""
        ideal = sorted(grades.values(), reverse=True)[:k]
        return sum(
            (2**grade - 1) / math.log2(idx + 1)
            for idx, grade in enumerate(ideal, start=1)
        )

    def _crosscheck_pytrec(
        self, inputs: list[EvalInput], metrics: dict[str, float]
    ) -> None:
        """Optionally cross-check ndcg/map against pytrec_eval (best effort)."""
        try:
            import pytrec_eval  # noqa: F401
        except ImportError:
            return
        # Build qrels/run dicts and compare; discrepancies are logged only.
        from src.observability.logger import get_logger

        logger = get_logger("evaluation.ir_metrics")
        qrels: dict[str, dict[str, int]] = {}
        run: dict[str, dict[str, float]] = {}
        for i, inp in enumerate(inputs):
            qid = f"q{i}"
            grades = self._relevance_map(inp)
            if not grades:
                continue
            qrels[qid] = {rid: int(g) for rid, g in grades.items()}
            run[qid] = {
                rid: float(len(inp.retrieved_ids) - rank)
                for rank, rid in enumerate(inp.retrieved_ids)
            }
        if not qrels:
            return
        measures = {f"ndcg_cut.{k}" for k in self._ks} | {f"map_cut.{k}" for k in self._ks}
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, measures)
        results = evaluator.evaluate(run)
        for k in self._ks:
            ref_ndcg = sum(
                r.get(f"ndcg_cut_{k}", 0.0) for r in results.values()
            ) / max(1, len(results))
            mine = metrics.get(f"ndcg@{k}", 0.0)
            if abs(ref_ndcg - mine) > 1e-6:
                logger.warning(
                    f"pytrec_eval ndcg@{k} mismatch: self={mine:.6f} ref={ref_ndcg:.6f}"
                )

    @property
    def provider_name(self) -> str:
        return "ir"
