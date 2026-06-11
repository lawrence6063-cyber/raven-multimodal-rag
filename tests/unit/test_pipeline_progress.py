"""Unit tests for IngestionPipeline.on_progress callback (F5)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

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
    pipe._batch_processor = MagicMock()
    pipe._batch_processor.process.return_value = [SimpleNamespace(id="doc1_0001")]
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


def test_on_progress_invoked_for_each_stage(doc_file):
    pipe = _mocked_pipeline()
    calls: list[tuple] = []

    pipe.run(doc_file, on_progress=lambda name, cur, total: calls.append((name, cur, total)))

    stage_names = {c[0] for c in calls}
    assert {
        "integrity_check",
        "load",
        "split",
        "transform",
        "encode",
        "store",
    } <= stage_names


def test_on_progress_transform_and_store_counts(doc_file):
    pipe = _mocked_pipeline()
    calls: list[tuple] = []

    pipe.run(doc_file, on_progress=lambda *a: calls.append(a))

    transform_calls = [c for c in calls if c[0] == "transform"]
    # transform reports 0..3 over total 3
    assert (("transform", 0, 3) in calls) and (("transform", 3, 3) in calls)
    store_calls = [c for c in calls if c[0] == "store"]
    assert (("store", 0, 2) in calls) and (("store", 2, 2) in calls)
    assert len(transform_calls) == 4


def test_on_progress_none_does_not_break(doc_file):
    pipe = _mocked_pipeline()
    result = pipe.run(doc_file)  # no on_progress
    assert result["status"] == "success"


def test_skip_path_returns_early(doc_file):
    pipe = _mocked_pipeline()
    pipe._integrity.should_skip.return_value = True
    calls: list[tuple] = []

    result = pipe.run(doc_file, on_progress=lambda *a: calls.append(a))

    assert result["status"] == "skipped"
    # Only the integrity_check progress fired before the early return
    assert calls == [("integrity_check", 0, 1)]
