"""Tests for RagasEvaluator (mock backend + graceful degradation)."""

import builtins

import pytest

from src.libs.evaluator.base_evaluator import EvalInput, EvaluatorError
from src.libs.evaluator.evaluator_factory import EvaluatorFactory
from src.observability.evaluation.ragas_evaluator import (
    DEFAULT_RAGAS_METRICS,
    RagasEvaluator,
)


def _sample_inputs():
    return [
        EvalInput(
            query="如何配置 Azure OpenAI？",
            retrieved_ids=["c1", "c2"],
            golden_ids=["c1"],
            retrieved_texts=["配置 endpoint 与 api_key", "设置 deployment"],
            answer="在 settings.yaml 中填写 endpoint 与 api_key。",
            contexts=["Azure OpenAI 需要 endpoint 与 api_key"],
        ),
    ]


class TestRagasEvaluator:
    """Test RagasEvaluator with an injected mock backend."""

    def test_evaluate_returns_metric_shape(self):
        def fake_backend(samples, metric_names):
            assert len(samples) == 1
            return {"faithfulness": 0.9, "answer_relevancy": 0.8, "context_precision": 0.7}

        evaluator = RagasEvaluator(backend=fake_backend)
        result = evaluator.evaluate(_sample_inputs())

        assert "faithfulness" in result.metrics
        assert "answer_relevancy" in result.metrics
        assert result.metrics["faithfulness"] == pytest.approx(0.9)
        assert result.metrics["answer_relevancy"] == pytest.approx(0.8)
        assert result.details["total_queries"] == 1

    def test_missing_metric_defaults_to_zero(self):
        def partial_backend(samples, metric_names):
            return {"faithfulness": 0.5}

        evaluator = RagasEvaluator(backend=partial_backend)
        result = evaluator.evaluate(_sample_inputs())

        assert result.metrics["faithfulness"] == pytest.approx(0.5)
        assert result.metrics["answer_relevancy"] == 0.0
        assert result.metrics["context_precision"] == 0.0

    def test_custom_metric_subset(self):
        evaluator = RagasEvaluator(
            metrics=("faithfulness",),
            backend=lambda s, m: {"faithfulness": 1.0},
        )
        result = evaluator.evaluate(_sample_inputs())
        assert set(result.metrics.keys()) == {"faithfulness"}

    def test_empty_inputs(self):
        evaluator = RagasEvaluator(backend=lambda s, m: {})
        result = evaluator.evaluate([])
        assert result.metrics == {name: 0.0 for name in DEFAULT_RAGAS_METRICS}
        assert result.details["total_queries"] == 0

    def test_backend_exception_wrapped(self):
        def broken_backend(samples, metric_names):
            raise RuntimeError("boom")

        evaluator = RagasEvaluator(backend=broken_backend)
        with pytest.raises(EvaluatorError, match="ragas evaluation failed"):
            evaluator.evaluate(_sample_inputs())

    def test_provider_name(self):
        assert RagasEvaluator().provider_name == "ragas"

    def test_missing_ragas_raises_import_error(self, monkeypatch):
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "ragas" or name.startswith("ragas."):
                raise ImportError("No module named 'ragas'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        evaluator = RagasEvaluator()  # no injected backend -> builds default
        with pytest.raises(EvaluatorError, match="Ragas is not installed"):
            evaluator.evaluate(_sample_inputs())


class TestFactoryRagasRegistration:
    """Ragas backend should be discoverable through the factory."""

    def test_ragas_registered(self):
        assert "ragas" in EvaluatorFactory.available_backends()

    def test_factory_creates_ragas(self):
        evaluator = EvaluatorFactory.create("ragas")
        assert isinstance(evaluator, RagasEvaluator)
