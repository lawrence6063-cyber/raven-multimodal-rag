"""Unit tests for JSON Lines trace logging (F2)."""

from __future__ import annotations

import json
import logging

from src.observability.logger import (
    JSONFormatter,
    get_logger,
    get_trace_logger,
    write_trace,
)


def test_json_formatter_dict_message():
    formatter = JSONFormatter()
    record = logging.LogRecord(
        name="t", level=logging.INFO, pathname="", lineno=0,
        msg={"trace_type": "query", "n": 1}, args=(), exc_info=None,
    )
    out = formatter.format(record)
    parsed = json.loads(out)
    assert parsed == {"trace_type": "query", "n": 1}


def test_json_formatter_string_message():
    formatter = JSONFormatter()
    record = logging.LogRecord(
        name="t", level=logging.INFO, pathname="", lineno=0,
        msg="hello %s", args=("world",), exc_info=None,
    )
    parsed = json.loads(formatter.format(record))
    assert parsed == {"message": "hello world"}


def test_write_trace_appends_one_json_line(tmp_path):
    log_file = str(tmp_path / "traces.jsonl")
    write_trace({"trace_id": "a", "trace_type": "query"}, log_file)
    write_trace({"trace_id": "b", "trace_type": "ingestion"}, log_file)

    lines = [l for l in open(log_file, encoding="utf-8").read().splitlines() if l]
    assert len(lines) == 2
    assert json.loads(lines[0])["trace_type"] == "query"
    assert json.loads(lines[1])["trace_type"] == "ingestion"


def test_trace_logger_is_singleton_per_file(tmp_path):
    log_file = str(tmp_path / "t.jsonl")
    a = get_trace_logger(log_file)
    b = get_trace_logger(log_file)
    assert a is b
    assert len(a.handlers) == 1
    assert a.propagate is False


def test_app_logger_unaffected_stays_on_stderr():
    logger = get_logger("phase_f_probe")
    # App logger should not write to any trace file handler
    assert all(not isinstance(h, logging.FileHandler) for h in logger.handlers)
