"""BaseEvaluator — abstract interface for RAG evaluation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvalInput:
    """Input for evaluation."""

    query: str
    retrieved_ids: list[str]
    golden_ids: list[str]
    retrieved_texts: list[str] = field(default_factory=list)
    answer: str = ""
    contexts: list[str] = field(default_factory=list)


@dataclass
class EvalResult:
    """Result from evaluation containing metrics."""

    metrics: dict[str, float] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)


class BaseEvaluator(ABC):
    """Abstract base class for evaluator implementations."""

    @abstractmethod
    def evaluate(self, inputs: list[EvalInput]) -> EvalResult:
        """Evaluate retrieval quality.

        Args:
            inputs: List of EvalInput objects with query + retrieved + golden data.

        Returns:
            EvalResult with aggregated metrics.

        Raises:
            EvaluatorError: If evaluation fails.
        """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the evaluator provider name."""


class EvaluatorError(Exception):
    """Raised when evaluation fails."""

    def __init__(self, message: str, provider: str = ""):
        self.provider = provider
        super().__init__(f"[{provider}] {message}" if provider else message)
