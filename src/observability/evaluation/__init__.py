"""Evaluation package — pluggable RAG quality evaluators and runner.

Exposes RagasEvaluator, CompositeEvaluator, EvalRunner and EvalReport.
Heavy dependencies (ragas) are imported lazily inside the evaluators, so
importing this package stays cheap and offline-friendly.
"""

from __future__ import annotations

from src.observability.evaluation.composite_evaluator import CompositeEvaluator
from src.observability.evaluation.eval_runner import EvalReport, EvalRunner
from src.observability.evaluation.ragas_evaluator import RagasEvaluator

__all__ = [
    "CompositeEvaluator",
    "EvalReport",
    "EvalRunner",
    "RagasEvaluator",
]
