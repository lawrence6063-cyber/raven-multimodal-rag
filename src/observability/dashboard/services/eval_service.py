"""EvalService — encapsulates evaluation runs for the Dashboard panel (H4)."""

from __future__ import annotations

from pathlib import Path

from src.core.query_engine.hybrid_search import HybridSearch
from src.core.settings import Settings
from src.libs.evaluator.evaluator_factory import EvaluatorFactory
from src.observability.evaluation.eval_runner import EvalReport, EvalRunner
from src.observability.logger import get_logger

logger = get_logger("dashboard.eval_service")


class EvalService:
    """Provides evaluation execution for the Dashboard evaluation panel."""

    def __init__(self, settings: Settings | None = None):
        self._settings = settings

    def _get_settings(self) -> Settings:
        if self._settings is None:
            from src.core.settings import load_settings

            self._settings = load_settings()
        return self._settings

    def available_backends(self) -> list[str]:
        """Return the evaluator backends registered in the factory."""
        return EvaluatorFactory.available_backends()

    def default_test_set(self) -> str:
        """Return the configured golden test set path."""
        return self._get_settings().evaluation.golden_test_set

    def run(self, backends: list[str], test_set_path: str | None = None) -> EvalReport:
        """Run an evaluation with the given backends over the test set.

        Args:
            backends: Evaluator backends to combine.
            test_set_path: Optional golden test set path override.

        Returns:
            The resulting :class:`EvalReport`.

        Raises:
            FileNotFoundError: If the test set file does not exist.
        """
        settings = self._get_settings()
        path = test_set_path or settings.evaluation.golden_test_set
        if not Path(path).exists():
            raise FileNotFoundError(f"Golden test set not found: {path}")

        evaluator = EvaluatorFactory.create_composite(backends)
        hybrid = HybridSearch(settings)
        runner = EvalRunner(settings, hybrid, evaluator)
        return runner.run(path)
