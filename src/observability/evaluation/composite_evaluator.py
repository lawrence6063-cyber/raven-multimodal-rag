"""CompositeEvaluator — runs multiple evaluators and merges their metrics.

Each wrapped evaluator runs concurrently (IO/LLM bound) via a thread pool.
A single evaluator failure is captured and degraded so the others still
produce results. Merged metric keys are prefixed by provider name to avoid
collisions, e.g. ``ragas.faithfulness`` and ``custom.hit_rate``.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from src.libs.evaluator.base_evaluator import BaseEvaluator, EvalInput, EvalResult
from src.observability.logger import get_logger

logger = get_logger("evaluation.composite_evaluator")


class CompositeEvaluator(BaseEvaluator):
    """Combine multiple evaluators, running them in parallel."""

    def __init__(self, evaluators: list[BaseEvaluator]):
        """Initialize with the evaluators to combine.

        Args:
            evaluators: Ordered list of evaluator instances.

        Raises:
            ValueError: If no evaluators are provided.
        """
        if not evaluators:
            raise ValueError("CompositeEvaluator requires at least one evaluator")
        self._evaluators = evaluators

    def evaluate(self, inputs: list[EvalInput]) -> EvalResult:
        """Run all evaluators concurrently and merge their metrics.

        Args:
            inputs: Evaluation samples shared across all evaluators.

        Returns:
            EvalResult whose metrics are prefixed by each provider name. Failed
            evaluators are recorded under ``details['errors']`` and skipped.
        """
        merged_metrics: dict[str, float] = {}
        details: dict[str, Any] = {"providers": [], "errors": {}}

        max_workers = max(1, len(self._evaluators))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(ev.evaluate, inputs): ev for ev in self._evaluators
            }
            for future, evaluator in futures.items():
                provider = evaluator.provider_name
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001 - degrade per evaluator
                    logger.warning(f"Evaluator '{provider}' failed: {exc}")
                    details["errors"][provider] = str(exc)
                    continue

                details["providers"].append(provider)
                for name, value in result.metrics.items():
                    merged_metrics[f"{provider}.{name}"] = value
                if result.details:
                    details[provider] = result.details

        return EvalResult(metrics=merged_metrics, details=details)

    @property
    def provider_name(self) -> str:
        names = "+".join(ev.provider_name for ev in self._evaluators)
        return f"composite({names})"
