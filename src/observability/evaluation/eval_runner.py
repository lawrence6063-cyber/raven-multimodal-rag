"""EvalRunner — runs the retrieval pipeline over a golden test set.

Loads a golden test set, executes hybrid search for each query, assembles
``EvalInput`` samples and delegates metric computation to a pluggable
evaluator. Produces an :class:`EvalReport` with aggregated metrics and
per-query details suitable for the dashboard and regression tests.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.libs.evaluator.base_evaluator import EvalInput
from src.observability.logger import get_logger

if TYPE_CHECKING:
    from src.core.query_engine.hybrid_search import HybridSearch
    from src.core.settings import Settings
    from src.core.types import RetrievalResult
    from src.libs.evaluator.base_evaluator import BaseEvaluator

logger = get_logger("evaluation.eval_runner")

# _SOURCE_KEYS metadata keys probed (in order) to resolve a chunk's source
_SOURCE_KEYS = ("source_path", "file_name", "source", "doc_id")


@dataclass
class EvalReport:
    """Aggregated evaluation report."""

    metrics: dict[str, float] = field(default_factory=dict)
    per_query: list[dict[str, Any]] = field(default_factory=list)
    backends: list[str] = field(default_factory=list)
    test_set_path: str = ""
    total_queries: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the report."""
        return asdict(self)


class EvalRunner:
    """Executes retrieval over a golden test set and evaluates the results."""

    def __init__(
        self,
        settings: "Settings",
        hybrid_search: "HybridSearch",
        evaluator: "BaseEvaluator",
    ):
        """Initialize the runner.

        Args:
            settings: Root configuration object.
            hybrid_search: Retrieval orchestrator used to answer each query.
            evaluator: Evaluator (or composite) computing the metrics.
        """
        self._settings = settings
        self._search = hybrid_search
        self._evaluator = evaluator

    def run(self, test_set_path: str | None = None) -> EvalReport:
        """Run the evaluation over the golden test set.

        Args:
            test_set_path: Path to the golden test set JSON. Defaults to
                ``settings.evaluation.golden_test_set``.

        Returns:
            An :class:`EvalReport` with aggregated metrics and per-query data.

        Raises:
            FileNotFoundError: If the test set file does not exist.
            ValueError: If the test set JSON is malformed.
        """
        path = test_set_path or self._settings.evaluation.golden_test_set
        test_cases = self._load_test_set(path)

        eval_inputs: list[EvalInput] = []
        per_query: list[dict[str, Any]] = []

        for case in test_cases:
            query = case.get("query", "")
            expected_chunk_ids = case.get("expected_chunk_ids") or []
            expected_sources = case.get("expected_sources") or []

            results = self._safe_search(query)
            retrieved_ids = [r.chunk_id for r in results]
            retrieved_sources = [self._resolve_source(r) for r in results]
            retrieved_texts = [r.text for r in results]

            # Choose the id space: prefer explicit chunk ids, fall back to sources.
            if expected_chunk_ids:
                ids = retrieved_ids
                golden = expected_chunk_ids
            else:
                ids = retrieved_sources
                golden = expected_sources

            eval_inputs.append(
                EvalInput(
                    query=query,
                    retrieved_ids=ids,
                    golden_ids=golden,
                    retrieved_texts=retrieved_texts,
                    contexts=retrieved_texts,
                )
            )

            hit = bool(set(ids) & set(golden))
            per_query.append(
                {
                    "query": query,
                    "retrieved_ids": retrieved_ids,
                    "retrieved_sources": retrieved_sources,
                    "expected_chunk_ids": expected_chunk_ids,
                    "expected_sources": expected_sources,
                    "num_retrieved": len(results),
                    "hit": hit,
                }
            )

        result = self._evaluator.evaluate(eval_inputs)
        backends = result.details.get("providers") if result.details else None

        return EvalReport(
            metrics=result.metrics,
            per_query=per_query,
            backends=backends or [self._evaluator.provider_name],
            test_set_path=str(path),
            total_queries=len(test_cases),
        )

    def _safe_search(self, query: str) -> list["RetrievalResult"]:
        """Run hybrid search, degrading to empty results on failure."""
        try:
            return self._search.search(query)
        except Exception as exc:  # noqa: BLE001 - one bad query must not abort the run
            logger.warning(f"Search failed for query '{query}': {exc}")
            return []

    @staticmethod
    def _load_test_set(path: str) -> list[dict[str, Any]]:
        """Load and validate the golden test set file."""
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Golden test set not found: {path}")

        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid golden test set JSON: {exc}") from exc

        test_cases = data.get("test_cases")
        if not isinstance(test_cases, list):
            raise ValueError("Golden test set must contain a 'test_cases' list")
        return test_cases

    @staticmethod
    def _resolve_source(result: "RetrievalResult") -> str:
        """Resolve a human-readable source for a retrieval result."""
        metadata = result.metadata or {}
        for key in _SOURCE_KEYS:
            value = metadata.get(key)
            if value:
                return str(value)
        return "unknown"
