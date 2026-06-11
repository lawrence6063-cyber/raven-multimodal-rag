"""TraceService — reads and parses traces.jsonl for Dashboard visualization (G5)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.observability.logger import get_logger

logger = get_logger("dashboard.trace_service")

_DEFAULT_TRACE_FILE = "logs/traces.jsonl"


class TraceService:
    """Reads and filters trace records from the JSON Lines file."""

    def __init__(self, log_file: str = _DEFAULT_TRACE_FILE):
        self._log_file = log_file

    def list_traces(
        self,
        trace_type: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List traces, optionally filtered by type, newest first.

        Args:
            trace_type: Filter by "query" or "ingestion"; None = all.
            limit: Maximum number of traces to return.

        Returns:
            List of trace dicts, newest first.
        """
        path = Path(self._log_file)
        if not path.exists():
            return []

        traces: list[dict[str, Any]] = []
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        if trace_type and record.get("trace_type") != trace_type:
                            continue
                        traces.append(record)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.warning(f"Failed to read traces: {e}")

        # Newest first (by started_at desc)
        traces.sort(key=lambda t: t.get("started_at", ""), reverse=True)
        return traces[:limit]

    def get_trace(self, trace_id: str) -> dict[str, Any] | None:
        """Get a single trace by ID.

        Args:
            trace_id: The trace_id to look up.

        Returns:
            Trace dict or None.
        """
        path = Path(self._log_file)
        if not path.exists():
            return None

        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        if record.get("trace_id") == trace_id:
                            return record
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.warning(f"Failed to read trace {trace_id}: {e}")
        return None

    def get_stage_breakdown(self, trace: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract stage breakdown for visualization.

        Args:
            trace: A trace dict with 'stages' list.

        Returns:
            List of {name, method, elapsed_ms, ...} dicts.
        """
        return trace.get("stages", [])
