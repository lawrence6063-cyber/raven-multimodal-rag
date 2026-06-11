"""Observability logging — stderr app logger and JSON Lines trace persistence.

``get_logger`` returns the standard application logger that writes to stderr
(keeping stdout clean for the MCP Stdio transport). ``write_trace`` /
``get_trace_logger`` (F2) persist structured trace records as JSON Lines to a
file, independently of the stderr logger.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any

# _DEFAULT_TRACE_FILE default JSON Lines destination for traces
_DEFAULT_TRACE_FILE = "logs/traces.jsonl"
# _TRACE_LOGGER_NAME logging channel dedicated to trace persistence
_TRACE_LOGGER_NAME = "rag_trace"


def get_logger(name: str = "rag_server") -> logging.Logger:
    """Get a configured logger that outputs to stderr."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


class JSONFormatter(logging.Formatter):
    """Formatter that emits each record's message as a single-line JSON object.

    If the log record's ``msg`` is already a dict, it is serialized directly;
    otherwise the rendered message string is wrapped as ``{"message": ...}``.
    """

    def format(self, record: logging.LogRecord) -> str:
        if isinstance(record.msg, dict):
            payload: Any = record.msg
        else:
            payload = {"message": record.getMessage()}
        return json.dumps(payload, ensure_ascii=False)


def get_trace_logger(log_file: str = _DEFAULT_TRACE_FILE) -> logging.Logger:
    """Get a singleton logger that writes JSON Lines to ``log_file``.

    Handlers are attached once per destination path so repeated calls do not
    duplicate output. This logger does not propagate to the root logger.
    """
    logger = logging.getLogger(f"{_TRACE_LOGGER_NAME}.{log_file}")
    if not logger.handlers:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(str(path), encoding="utf-8")
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def write_trace(trace_dict: dict[str, Any], log_file: str = _DEFAULT_TRACE_FILE) -> None:
    """Append one trace record as a JSON line to ``log_file``.

    Args:
        trace_dict: A JSON-serializable trace dict (e.g. ``TraceContext.to_dict()``).
        log_file: Destination JSON Lines file path.
    """
    get_trace_logger(log_file).info(trace_dict)
