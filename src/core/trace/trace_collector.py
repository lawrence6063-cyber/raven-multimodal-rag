"""TraceCollector — persists finished traces when tracing is enabled (F1/F2).

Bridges TraceContext (F1) and the JSON Lines writer (F2). Collection failures
are swallowed and logged so observability never breaks the main flow.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.observability.logger import get_logger, write_trace

if TYPE_CHECKING:
    from src.core.settings import Settings
    from src.core.trace.trace_context import TraceContext

logger = get_logger("trace.collector")


class TraceCollector:
    """Collects traces and persists them to the configured JSON Lines file."""

    def __init__(self, settings: "Settings"):
        self._enabled = settings.observability.trace_enabled
        self._log_file = settings.observability.log_file

    def collect(self, trace: "TraceContext") -> None:
        """Persist a trace if tracing is enabled; never raises.

        Args:
            trace: A finished TraceContext to persist.
        """
        if not self._enabled:
            return
        try:
            write_trace(trace.to_dict(), self._log_file)
        except Exception as e:  # observability must not break the main flow
            logger.warning(f"Failed to persist trace {trace.trace_id}: {e}")
