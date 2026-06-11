"""Tests for CompositeEvaluator (parallel merge + per-evaluator degradation)."""

import pytest

from src.libs.evaluator.base_evaluator import (
    BaseEvaluator,
    EvalInput,
    EvalResult,
)
from src.libs.evaluator.custom_evaluator import CustomEvaluator
from src.libs.evaluator.evaluator_factory import EvaluatorFactory
from src.observability.evaluation.composite_evaluator import CompositeEvaluator
from src.observability.evaluation.ragas_evaluator import RagasEvaluator


class _StubEvaluator(BaseEvaluator):
    def __init__(self, name, metrics):
        self._name = name
        self._metrics = metrics

    def evaluate(self, inputs):
        return EvalResult(metrics=dict(self._metrics), details={"n": len(inputs)})

    @property
    def provider_name(self):
        return self._name


class _BrokenEvaluator(BaseEvaluator):
    def evaluate(self, inputs):
        raise RuntimeError("broken backend")

    @property
    def provider_name(self):
        return "broken"


def _inputs():
    return [EvalInput(query="q", retrieved_ids=["a", "b"], golden_ids=["a"])]


class TestCompositeEvaluator:
    """Test CompositeEvaluator metric merging."""

    def test_requires_at_least_one_evaluator(self):
        with pytest.raises(ValueError, match="at least one evaluator"):
            CompositeEvaluator([])

    def test_merges_metrics_with_provider_prefix(self):
        composite = CompositeEvaluator(
            [
                _StubEvaluator("alpha", {"hit_rate": 1.0}),
                _StubEvaluator("beta", {"faithfulness": 0.5}),
            ]
        )
        result = composite.evaluate(_inputs())

        assert result.metrics["alpha.hit_rate"] == pytest.approx(1.0)
        assert result.metrics["beta.faithfulness"] == pytest.approx(0.5)
        assert set(result.details["providers"]) == {"alpha", "beta"}

    def test_combines_real_custom_and_ragas(self):
        composite = CompositeEvaluator(
            [
                CustomEvaluator(),
                RagasEvaluator(backend=lambda s, m: {"faithfulness": 0.9}),
            ]
        )
        result = composite.evaluate(_inputs())

        assert "custom.hit_rate" in result.metrics
        assert "ragas.faithfulness" in result.metrics

    def test_single_failure_does_not_block_others(self):
        composite = CompositeEvaluator(
            [
                _StubEvaluator("ok", {"score": 1.0}),
                _BrokenEvaluator(),
            ]
        )
        result = composite.evaluate(_inputs())

        assert result.metrics["ok.score"] == pytest.approx(1.0)
        assert "broken" in result.details["errors"]
        assert "broken backend" in result.details["errors"]["broken"]

    def test_provider_name(self):
        composite = CompositeEvaluator([CustomEvaluator()])
        assert composite.provider_name == "composite(custom)"


class TestFactoryComposite:
    """Test EvaluatorFactory.create_composite."""

    def test_create_composite_from_backends(self):
        composite = EvaluatorFactory.create_composite(["custom", "ragas"])
        assert isinstance(composite, CompositeEvaluator)
        result = composite.evaluate(_inputs())
        assert "custom.hit_rate" in result.metrics

    def test_create_composite_empty_raises(self):
        from src.libs.evaluator.base_evaluator import EvaluatorError
        with pytest.raises(EvaluatorError, match="No evaluator backends"):
            EvaluatorFactory.create_composite([])
