"""RagasEvaluator — wraps the Ragas framework as a pluggable BaseEvaluator.

Ragas is a heavy, LLM-backed dependency. To keep the rest of the system
offline-friendly and unit-testable, the actual ragas import happens lazily
inside the evaluation backend, and the metric-computation step is delegated to
an injectable callable so tests can supply a mock backend.
"""

from __future__ import annotations

from typing import Callable

from src.libs.evaluator.base_evaluator import (
    BaseEvaluator,
    EvalInput,
    EvalResult,
    EvaluatorError,
)
from src.libs.evaluator.evaluator_factory import register_evaluator
from src.observability.logger import get_logger

logger = get_logger("evaluation.ragas_evaluator")

# DEFAULT_RAGAS_METRICS 默认启用的 Ragas 指标名
DEFAULT_RAGAS_METRICS = ("faithfulness", "answer_relevancy", "context_precision")

# RagasBackend ragas 计算后端签名：输入样本与指标名，返回标准化指标字典
RagasBackend = Callable[[list[EvalInput], tuple[str, ...]], dict[str, float]]


@register_evaluator("ragas")
class RagasEvaluator(BaseEvaluator):
    """Evaluator backed by the Ragas framework.

    Supports Faithfulness, Answer Relevancy and Context Precision metrics.
    """

    def __init__(
        self,
        metrics: tuple[str, ...] | None = None,
        backend: RagasBackend | None = None,
    ):
        """Initialize the evaluator.

        Args:
            metrics: Ragas metric names to compute. Defaults to
                DEFAULT_RAGAS_METRICS.
            backend: Optional injectable computation backend. When omitted, a
                ragas-backed backend is built lazily on first use. Tests can
                inject a mock to avoid real LLM calls.
        """
        self._metrics = tuple(metrics) if metrics else DEFAULT_RAGAS_METRICS
        self._backend = backend

    def evaluate(self, inputs: list[EvalInput]) -> EvalResult:
        """Compute ragas metrics over the given evaluation inputs.

        Args:
            inputs: Evaluation samples carrying query/answer/contexts.

        Returns:
            EvalResult with the aggregated ragas metrics.

        Raises:
            EvaluatorError: If ragas is unavailable or computation fails.
        """
        if not inputs:
            return EvalResult(
                metrics={name: 0.0 for name in self._metrics},
                details={"total_queries": 0},
            )

        backend = self._backend or self._build_default_backend()
        try:
            raw_metrics = backend(inputs, self._metrics)
        except EvaluatorError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalize into EvaluatorError
            raise EvaluatorError(f"ragas evaluation failed: {exc}", provider="ragas") from exc

        metrics = {name: float(raw_metrics.get(name, 0.0)) for name in self._metrics}
        return EvalResult(
            metrics=metrics,
            details={"total_queries": len(inputs), "metric_names": list(self._metrics)},
        )

    @property
    def provider_name(self) -> str:
        return "ragas"

    def _build_default_backend(self) -> RagasBackend:
        """Build the default ragas-backed computation function (lazy import).

        Raises:
            EvaluatorError: If the ragas package is not installed.
        """
        try:
            import ragas  # noqa: F401
        except ImportError as exc:
            raise EvaluatorError(
                "Ragas is not installed. Install it with `pip install ragas` "
                "to enable the ragas evaluator backend.",
                provider="ragas",
            ) from exc

        def _backend(samples: list[EvalInput], metric_names: tuple[str, ...]) -> dict[str, float]:
            return self._run_ragas(samples, metric_names)

        return _backend

    @staticmethod
    def _run_ragas(samples: list[EvalInput], metric_names: tuple[str, ...]) -> dict[str, float]:
        """Run real ragas evaluation over the samples.

        This path requires ragas (and a configured LLM/embeddings) and is not
        exercised by the offline unit tests. It builds a ragas dataset from the
        EvalInput samples and aggregates the requested metric scores.
        """
        from datasets import Dataset
        from ragas import evaluate as ragas_evaluate

        rows = {
            "question": [s.query for s in samples],
            "answer": [s.answer for s in samples],
            "contexts": [s.contexts or s.retrieved_texts for s in samples],
            "ground_truth": ["\n".join(s.contexts) for s in samples],
        }
        dataset = Dataset.from_dict(rows)
        result = ragas_evaluate(dataset)
        scores = result.to_pandas().mean(numeric_only=True).to_dict()
        return {name: float(scores.get(name, 0.0)) for name in metric_names}
