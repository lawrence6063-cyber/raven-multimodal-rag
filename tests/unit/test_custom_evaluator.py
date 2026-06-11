"""Tests for CustomEvaluator (hit_rate/mrr)."""

import pytest

from src.libs.evaluator.base_evaluator import EvalInput, EvalResult
from src.libs.evaluator.custom_evaluator import CustomEvaluator
from src.libs.evaluator.evaluator_factory import EvaluatorFactory, _EVALUATOR_REGISTRY


class TestCustomEvaluator:
    """Test CustomEvaluator metric computation."""

    def test_perfect_hit_rate(self):
        evaluator = CustomEvaluator()
        inputs = [
            EvalInput(query="q1", retrieved_ids=["a", "b", "c"], golden_ids=["a"]),
            EvalInput(query="q2", retrieved_ids=["x", "y"], golden_ids=["y"]),
        ]
        result = evaluator.evaluate(inputs)
        assert result.metrics["hit_rate"] == 1.0

    def test_zero_hit_rate(self):
        evaluator = CustomEvaluator()
        inputs = [
            EvalInput(query="q1", retrieved_ids=["a", "b"], golden_ids=["z"]),
            EvalInput(query="q2", retrieved_ids=["x", "y"], golden_ids=["w"]),
        ]
        result = evaluator.evaluate(inputs)
        assert result.metrics["hit_rate"] == 0.0
        assert result.metrics["mrr"] == 0.0

    def test_partial_hit_rate(self):
        evaluator = CustomEvaluator()
        inputs = [
            EvalInput(query="q1", retrieved_ids=["a", "b"], golden_ids=["a"]),
            EvalInput(query="q2", retrieved_ids=["x", "y"], golden_ids=["z"]),
        ]
        result = evaluator.evaluate(inputs)
        assert result.metrics["hit_rate"] == 0.5

    def test_mrr_computation(self):
        evaluator = CustomEvaluator()
        inputs = [
            EvalInput(query="q1", retrieved_ids=["a", "b", "c"], golden_ids=["a"]),  # rank 1 -> 1/1
            EvalInput(query="q2", retrieved_ids=["x", "y", "z"], golden_ids=["z"]),  # rank 3 -> 1/3
        ]
        result = evaluator.evaluate(inputs)
        expected_mrr = (1.0 + 1.0 / 3.0) / 2.0
        assert abs(result.metrics["mrr"] - expected_mrr) < 1e-6

    def test_empty_inputs(self):
        evaluator = CustomEvaluator()
        result = evaluator.evaluate([])
        assert result.metrics["hit_rate"] == 0.0
        assert result.metrics["mrr"] == 0.0

    def test_provider_name(self):
        assert CustomEvaluator().provider_name == "custom"

    def test_eval_result_has_details(self):
        evaluator = CustomEvaluator()
        inputs = [EvalInput(query="q", retrieved_ids=["a"], golden_ids=["a"])]
        result = evaluator.evaluate(inputs)
        assert result.details["total_queries"] == 1
        assert result.details["hits"] == 1


class TestCustomEvaluatorBoundaries:
    """Boundary contract for CustomEvaluator inputs."""

    def test_empty_golden_ids_counts_as_miss(self):
        evaluator = CustomEvaluator()
        inputs = [EvalInput(query="q", retrieved_ids=["a", "b"], golden_ids=[])]
        result = evaluator.evaluate(inputs)
        assert result.metrics["hit_rate"] == 0.0
        assert result.metrics["mrr"] == 0.0

    def test_empty_retrieved_ids_counts_as_miss(self):
        evaluator = CustomEvaluator()
        inputs = [EvalInput(query="q", retrieved_ids=[], golden_ids=["a"])]
        result = evaluator.evaluate(inputs)
        assert result.metrics["hit_rate"] == 0.0
        assert result.details["hits"] == 0

    def test_multiple_golden_ids_hits_on_any(self):
        evaluator = CustomEvaluator()
        inputs = [
            EvalInput(query="q", retrieved_ids=["x", "b", "c"], golden_ids=["a", "b"]),
        ]
        result = evaluator.evaluate(inputs)
        assert result.metrics["hit_rate"] == 1.0
        # First relevant ("b") is at rank 2 -> mrr 0.5.
        assert abs(result.metrics["mrr"] - 0.5) < 1e-9

    def test_mrr_uses_first_match_rank_only(self):
        evaluator = CustomEvaluator()
        inputs = [
            EvalInput(query="q", retrieved_ids=["a", "a", "a"], golden_ids=["a"]),
        ]
        result = evaluator.evaluate(inputs)
        assert result.metrics["mrr"] == 1.0

    def test_details_count_mixed_hits(self):
        evaluator = CustomEvaluator()
        inputs = [
            EvalInput(query="q1", retrieved_ids=["a"], golden_ids=["a"]),
            EvalInput(query="q2", retrieved_ids=["a"], golden_ids=["z"]),
            EvalInput(query="q3", retrieved_ids=["c"], golden_ids=["c"]),
        ]
        result = evaluator.evaluate(inputs)
        assert result.details["total_queries"] == 3
        assert result.details["hits"] == 2


class TestEvaluatorFactory:
    """Test EvaluatorFactory with CustomEvaluator."""

    def test_create_custom_evaluator(self):
        evaluator = EvaluatorFactory.create("custom")
        assert isinstance(evaluator, CustomEvaluator)

    def test_create_unknown_backend_raises(self):
        from src.libs.evaluator.base_evaluator import EvaluatorError
        with pytest.raises(EvaluatorError, match="Unknown evaluator backend"):
            EvaluatorFactory.create("nonexistent")

    def test_available_backends(self):
        backends = EvaluatorFactory.available_backends()
        assert "custom" in backends
