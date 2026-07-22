"""Unit tests for EvalRunner — answer-synthesis link and latency percentiles.

Hermetic (no network / no LLM): a deterministic fake search feeds the runner and
an injectable fake synthesizer exercises the optional generation-side answer
link introduced for the Ragas backend (EVAL_OPTIMIZATION_SPEC §4). Latency
percentiles (p50/p95/p99) and per-query ``synthesize_ms`` are also asserted.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.core.settings import Settings
from src.core.types import RetrievalResult
from src.libs.evaluator.base_evaluator import BaseEvaluator, EvalInput, EvalResult
from src.observability.evaluation.eval_runner import EvalRunner


class _FakeSearch:
    """Returns a fixed ranked list regardless of the query."""

    def search(self, query, top_k=None, filters=None, trace=None):
        return [
            RetrievalResult(chunk_id="c1", score=0.9, text="alpha passage", metadata={}),
            RetrievalResult(chunk_id="c2", score=0.8, text="beta passage", metadata={}),
        ]


class _CapturingEvaluator(BaseEvaluator):
    """Stores the inputs it receives so tests can inspect propagated fields."""

    def __init__(self):
        self.seen: list[EvalInput] = []

    def evaluate(self, inputs: list[EvalInput]) -> EvalResult:
        self.seen = list(inputs)
        return EvalResult(metrics={"hit_rate": 1.0}, details={})

    @property
    def provider_name(self) -> str:
        return "capturing"


class _FakeSynth:
    """Synthesizer returning an object with an ``answer`` attribute."""

    class _Out:
        def __init__(self, answer: str):
            self.answer = answer

    def answer(self, query, context):
        return self._Out(f"answer for: {query}")


class _RaisingSynth:
    def answer(self, query, context):
        raise RuntimeError("boom")


def _write_golden(tmp_path: Path) -> str:
    data = {
        "test_cases": [
            {"query": "q1", "expected_chunk_ids": ["c1"], "relevance": {"c1": 2}},
            {"query": "q2", "expected_chunk_ids": ["c2"], "relevance": {"c2": 1}},
        ]
    }
    path = tmp_path / "golden.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


class TestAnswerSynthesisLink:
    def test_no_synthesizer_leaves_answer_empty(self, tmp_path):
        evaluator = _CapturingEvaluator()
        runner = EvalRunner(Settings(), _FakeSearch(), evaluator)
        report = runner.run(_write_golden(tmp_path))

        assert all(inp.answer == "" for inp in evaluator.seen)
        assert all(row["synthesize_ms"] == 0.0 for row in report.per_query)

    def test_synthesizer_populates_answer(self, tmp_path):
        evaluator = _CapturingEvaluator()
        runner = EvalRunner(
            Settings(), _FakeSearch(), evaluator, answer_synthesizer=_FakeSynth()
        )
        runner.run(_write_golden(tmp_path))

        answers = [inp.answer for inp in evaluator.seen]
        assert answers == ["answer for: q1", "answer for: q2"]

    def test_synthesis_failure_degrades_to_empty(self, tmp_path):
        evaluator = _CapturingEvaluator()
        runner = EvalRunner(
            Settings(), _FakeSearch(), evaluator, answer_synthesizer=_RaisingSynth()
        )
        # Must not raise; answer degrades to "".
        runner.run(_write_golden(tmp_path))
        assert all(inp.answer == "" for inp in evaluator.seen)


class TestLatencyPercentiles:
    def test_p50_p95_p99_present(self, tmp_path):
        runner = EvalRunner(Settings(), _FakeSearch(), _CapturingEvaluator())
        report = runner.run(_write_golden(tmp_path))
        for key in ("avg_latency_ms", "p50_latency_ms", "p95_latency_ms", "p99_latency_ms"):
            assert key in report.metrics

    def test_percentile_ordering(self):
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        p50 = EvalRunner._percentile(values, 50)
        p95 = EvalRunner._percentile(values, 95)
        p99 = EvalRunner._percentile(values, 99)
        assert p50 <= p95 <= p99
        assert p50 == 30.0
