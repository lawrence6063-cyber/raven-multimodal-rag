"""CustomEvaluator — lightweight hit_rate/mrr metrics implementation."""

from __future__ import annotations

from src.libs.evaluator.base_evaluator import BaseEvaluator, EvalInput, EvalResult
from src.libs.evaluator.evaluator_factory import register_evaluator


@register_evaluator("custom")
class CustomEvaluator(BaseEvaluator):
    """Custom evaluator computing hit_rate and MRR metrics."""

    def evaluate(self, inputs: list[EvalInput]) -> EvalResult:
        """Compute hit_rate, MRR, and recall_completeness over the evaluation inputs.

        hit_rate: fraction of queries where at least one golden ID is in retrieved.
        mrr: Mean Reciprocal Rank — average of 1/rank of first relevant result.
        recall_completeness: average capped recall — |retrieved ∩ golden| /
            min(|retrieved|, |golden|). The capped denominator avoids the
            假性低分 that raw |golden| produces when the relevant set is far
            larger than the number of retrieved slots (EVAL_OPTIMIZATION_SPEC
            §3.1.4).
        """
        if not inputs:
            return EvalResult(metrics={"hit_rate": 0.0, "mrr": 0.0, "recall_completeness": 0.0})

        hits = 0
        reciprocal_ranks = []
        recall_scores = []

        for inp in inputs:
            golden_set = set(inp.golden_ids)
            retrieved_set = set(inp.retrieved_ids)

            # Hit Rate & MRR (existing logic)
            found_rank = 0
            for rank, rid in enumerate(inp.retrieved_ids, start=1):
                if rid in golden_set:
                    found_rank = rank
                    break

            if found_rank > 0:
                hits += 1
                reciprocal_ranks.append(1.0 / found_rank)
            else:
                reciprocal_ranks.append(0.0)

            # Recall Completeness (capped recall, spec §3.1.4)
            denom = min(len(retrieved_set), len(golden_set))
            if denom > 0:
                recall = len(retrieved_set & golden_set) / denom
            else:
                recall = 0.0
            recall_scores.append(recall)

        n = len(inputs)
        hit_rate = hits / n
        mrr = sum(reciprocal_ranks) / n
        recall_completeness = sum(recall_scores) / n

        return EvalResult(
            metrics={"hit_rate": hit_rate, "mrr": mrr, "recall_completeness": recall_completeness},
            details={"total_queries": n, "hits": hits},
        )

    @property
    def provider_name(self) -> str:
        return "custom"
