"""Unit tests for Ingestion-chain tracing (F4)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.core.trace.trace_context import TraceContext
from src.core.types import Document
from src.ingestion.pipeline import IngestionPipeline


def _mocked_pipeline() -> IngestionPipeline:
    """Build an IngestionPipeline with all components mocked (no IO/network)."""
    pipe = IngestionPipeline.__new__(IngestionPipeline)
    pipe._settings = SimpleNamespace(
        embedding=SimpleNamespace(provider="openai"),
        splitter=SimpleNamespace(provider="recursive"),
    )
    pipe._integrity = MagicMock()
    pipe._integrity.compute_sha256.return_value = "hash123"
    pipe._integrity.should_skip.return_value = False

    pipe._loader = MagicMock()
    pipe._loader.load.return_value = Document(id="doc1", text="body", metadata={})

    chunk = SimpleNamespace(id="doc1_0001", text="body", metadata={})
    pipe._chunker = MagicMock()
    pipe._chunker.split_document.return_value = [chunk]
    pipe._refiner = MagicMock(); pipe._refiner.transform.side_effect = lambda c: c
    pipe._enricher = MagicMock(); pipe._enricher.transform.side_effect = lambda c: c
    pipe._captioner = MagicMock(); pipe._captioner.transform.side_effect = lambda c: c

    record = SimpleNamespace(id="doc1_0001")
    pipe._batch_processor = MagicMock()
    pipe._batch_processor.process.return_value = [record]
    pipe._image_encoder = MagicMock()
    pipe._image_encoder.encode_document.return_value = []
    pipe._upserter = MagicMock()
    pipe._bm25 = MagicMock()
    pipe._collector = MagicMock()
    return pipe


@pytest.fixture
def doc_file(tmp_path):
    f = tmp_path / "doc.pdf"
    f.write_text("dummy")
    return str(f)


def test_run_records_ingestion_stages(doc_file):
    pipe = _mocked_pipeline()
    trace = TraceContext(trace_type="ingestion")

    result = pipe.run(doc_file, collection="kb", trace=trace)

    assert result["status"] == "success"
    stage_names = [s["name"] for s in trace.stages]
    assert stage_names == ["load", "split", "transform", "embed", "upsert"]
    assert trace.trace_type == "ingestion"


def test_stage_details_and_methods(doc_file):
    pipe = _mocked_pipeline()
    trace = TraceContext(trace_type="ingestion")

    pipe.run(doc_file, trace=trace)

    by_name = {s["name"]: s for s in trace.stages}
    assert by_name["load"]["method"] == "markitdown"
    assert by_name["split"]["method"] == "recursive"
    assert by_name["split"]["chunks"] == 1
    assert by_name["embed"]["method"] == "openai"
    assert by_name["upsert"]["method"] == "chroma+bm25"
    for stage in trace.stages:
        assert "elapsed_ms" in stage


def test_run_creates_and_collects_trace_when_none(doc_file):
    pipe = _mocked_pipeline()

    pipe.run(doc_file)  # trace=None -> internal trace created

    pipe._collector.collect.assert_called_once()
    collected = pipe._collector.collect.call_args[0][0]
    assert collected.trace_type == "ingestion"
    assert collected.to_dict()["finished_at"] is not None
