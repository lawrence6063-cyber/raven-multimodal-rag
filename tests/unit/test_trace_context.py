"""Unit tests for TraceContext and TraceCollector (F1)."""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest

from src.core.trace.trace_collector import TraceCollector
from src.core.trace.trace_context import TraceContext


class TestTraceContext:
    def test_defaults_to_query_type(self):
        trace = TraceContext()
        assert trace.trace_type == "query"
        assert trace.trace_id

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError):
            TraceContext(trace_type="bogus")

    def test_record_stage_appends(self):
        trace = TraceContext(trace_type="ingestion")
        trace.record_stage("load", method="markitdown", elapsed_ms=12.5, pages=3)
        assert len(trace.stages) == 1
        stage = trace.stages[0]
        assert stage["name"] == "load"
        assert stage["method"] == "markitdown"
        assert stage["elapsed_ms"] == 12.5
        assert stage["pages"] == 3

    def test_stage_context_manager_times_block(self):
        trace = TraceContext()
        with trace.stage("dense_retrieval", method="openai") as extra:
            extra["hits"] = 5
            time.sleep(0.01)
        stage = trace.stages[0]
        assert stage["name"] == "dense_retrieval"
        assert stage["method"] == "openai"
        assert stage["hits"] == 5
        assert stage["elapsed_ms"] > 0

    def test_finish_freezes_total(self):
        trace = TraceContext()
        trace.finish()
        first = trace.elapsed_ms()
        time.sleep(0.01)
        assert trace.elapsed_ms() == first  # frozen after finish

    def test_elapsed_ms_for_named_stage(self):
        trace = TraceContext()
        trace.record_stage("fusion", elapsed_ms=7.0)
        assert trace.elapsed_ms("fusion") == 7.0
        assert trace.elapsed_ms("missing") == 0.0

    def test_to_dict_has_required_fields_and_is_json_serializable(self):
        trace = TraceContext(trace_type="query")
        trace.record_stage("query_processing", method="regex")
        trace.finish()
        d = trace.to_dict()
        assert {
            "trace_id",
            "trace_type",
            "started_at",
            "finished_at",
            "total_elapsed_ms",
            "stages",
        } <= set(d.keys())
        assert d["trace_type"] == "query"
        # Must be directly serializable
        json.dumps(d)


class TestTraceCollector:
    def _settings(self, tmp_path, enabled=True):
        log_file = str(tmp_path / "traces.jsonl")
        return SimpleNamespace(
            observability=SimpleNamespace(trace_enabled=enabled, log_file=log_file)
        ), log_file

    def test_collect_writes_when_enabled(self, tmp_path):
        settings, log_file = self._settings(tmp_path, enabled=True)
        trace = TraceContext(trace_type="ingestion")
        trace.record_stage("load", method="markitdown")
        trace.finish()

        TraceCollector(settings).collect(trace)

        lines = [l for l in open(log_file, encoding="utf-8").read().splitlines() if l]
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["trace_type"] == "ingestion"

    def test_collect_noop_when_disabled(self, tmp_path):
        settings, log_file = self._settings(tmp_path, enabled=False)
        trace = TraceContext()
        trace.finish()

        TraceCollector(settings).collect(trace)

        import os

        assert not os.path.exists(log_file)
