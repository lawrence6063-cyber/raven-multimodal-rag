"""Unit tests for scripts/merge_testset.py — dedup + richness conflict rules."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "merge_testset.py"
_spec = importlib.util.spec_from_file_location("merge_testset", _SCRIPT)
merge_testset = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(merge_testset)  # type: ignore[union-attr]


def _write(path: Path, cases: list[dict]) -> str:
    path.write_text(json.dumps({"test_cases": cases}), encoding="utf-8")
    return str(path)


class TestNormalize:
    def test_strip_lower_fold(self):
        assert merge_testset._normalize_query("  Hello   World ") == "hello world"

    def test_case_insensitive_dedup_key(self):
        assert merge_testset._normalize_query("RAG Test") == merge_testset._normalize_query("rag  test")


class TestRichness:
    def test_chunk_ids_beats_sources_only(self):
        rich = {"expected_chunk_ids": ["c1"], "relevance": {"c1": 3}}
        poor = {"expected_sources": ["s.pdf"]}
        assert merge_testset._richness(rich) > merge_testset._richness(poor)


class TestMergeCases:
    def test_dedup_keeps_richer(self, tmp_path):
        f1 = _write(tmp_path / "a.json", [
            {"query": "What is RAG?", "expected_sources": ["rag.pdf"], "source": "handwritten"},
        ])
        f2 = _write(tmp_path / "b.json", [
            {
                "query": "what is rag?",  # same after normalize
                "expected_chunk_ids": ["c1"],
                "relevance": {"c1": 3},
                "source": "llamaindex_synth",
            },
        ])
        merged, stats = merge_testset.merge_cases([f1, f2])
        assert len(merged) == 1
        # richer (chunk-level) entry wins
        assert merged[0]["expected_chunk_ids"] == ["c1"]
        assert stats == {"llamaindex_synth": 1}

    def test_distinct_queries_preserved_in_order(self, tmp_path):
        f1 = _write(tmp_path / "a.json", [
            {"query": "Q1", "source": "handwritten"},
            {"query": "Q2", "source": "handwritten"},
        ])
        f2 = _write(tmp_path / "b.json", [
            {"query": "Q3", "source": "ragas_synth"},
        ])
        merged, stats = merge_testset.merge_cases([f1, f2])
        assert [c["query"] for c in merged] == ["Q1", "Q2", "Q3"]
        assert stats == {"handwritten": 2, "ragas_synth": 1}

    def test_empty_query_skipped(self, tmp_path):
        f1 = _write(tmp_path / "a.json", [
            {"query": "   ", "source": "x"},
            {"query": "Real", "source": "handwritten"},
        ])
        merged, _ = merge_testset.merge_cases([f1])
        assert len(merged) == 1
        assert merged[0]["query"] == "Real"

    def test_missing_test_cases_raises(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"foo": 1}), encoding="utf-8")
        with pytest.raises(ValueError):
            merge_testset.merge_cases([str(bad)])
