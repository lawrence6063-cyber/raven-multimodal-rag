"""TraceContext — per-execution observability record for query/ingestion (F1).

A TraceContext accumulates timed stages over a single Query or Ingestion run.
Each stage captures its name, the method/provider used, elapsed time, and a few
lightweight detail fields (counts, flags) — never secrets or raw payloads.

Typical usage::

    trace = TraceContext(trace_type="query")
    with trace.stage("dense_retrieval", method="openai"):
        ...                          # timed automatically
    trace.record_stage("fusion", method="rrf", candidates=12)
    trace.finish()
    collector.collect(trace)
"""

from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

# _VALID_TYPES the allowed trace_type values
_VALID_TYPES = ("query", "ingestion")


def _now_ms() -> float:
    """Return the current wall-clock time in epoch milliseconds."""
    return time.time() * 1000.0


def _iso(ms: float) -> str:
    """Format epoch milliseconds as an ISO 8601 UTC string."""
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).isoformat()


class TraceContext:
    """Accumulates timed stages for one query or ingestion execution."""

    def __init__(self, trace_type: str = "query"):
        if trace_type not in _VALID_TYPES:
            raise ValueError(
                f"Invalid trace_type: {trace_type!r}; expected one of {_VALID_TYPES}"
            )
        self.trace_id: str = uuid.uuid4().hex
        self.trace_type: str = trace_type
        self._started_ms: float = _now_ms()
        self._finished_ms: float | None = None
        self.stages: list[dict[str, Any]] = []

    def record_stage(
        self,
        name: str,
        method: str = "",
        elapsed_ms: float | None = None,
        **details: Any,
    ) -> None:
        """Append a stage record.

        Args:
            name: Stage name (e.g. "dense_retrieval").
            method: Method/provider used (e.g. "openai", "rrf").
            elapsed_ms: Stage duration in milliseconds; 0.0 when unknown.
            **details: Extra lightweight, JSON-serializable detail fields.
        """
        stage: dict[str, Any] = {
            "name": name,
            "method": method,
            "elapsed_ms": round(float(elapsed_ms), 3) if elapsed_ms is not None else 0.0,
        }
        if details:
            stage.update(details)
        self.stages.append(stage)

    @contextmanager
    def stage(self, name: str, method: str = "", **details: Any) -> Iterator[dict[str, Any]]:
        """Context manager that times the wrapped block and records a stage.

        The yielded dict may be mutated to attach additional detail fields that
        are only known inside the block (e.g. result counts).
        """
        extra: dict[str, Any] = dict(details)
        start = time.perf_counter()
        try:
            yield extra
        finally:
            elapsed = (time.perf_counter() - start) * 1000.0
            self.record_stage(name, method=method, elapsed_ms=elapsed, **extra)

    def finish(self) -> None:
        """Mark the trace as finished and freeze the total elapsed time."""
        if self._finished_ms is None:
            self._finished_ms = _now_ms()

    def elapsed_ms(self, stage_name: str | None = None) -> float:
        """Return elapsed milliseconds for a stage, or the total when omitted.

        Args:
            stage_name: A stage name; when None, returns the total run time.

        Returns:
            Elapsed milliseconds. Returns 0.0 if the stage is not found.
        """
        if stage_name is not None:
            for stage in self.stages:
                if stage["name"] == stage_name:
                    return float(stage["elapsed_ms"])
            return 0.0

        end = self._finished_ms if self._finished_ms is not None else _now_ms()
        return round(end - self._started_ms, 3)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a fully JSON-serializable dict."""
        return {
            "trace_id": self.trace_id,
            "trace_type": self.trace_type,
            "started_at": _iso(self._started_ms),
            "finished_at": _iso(self._finished_ms) if self._finished_ms is not None else None,
            "total_elapsed_ms": self.elapsed_ms(),
            "stages": list(self.stages),
        }
