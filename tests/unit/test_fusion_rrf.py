"""Tests for RRF Fusion."""

import pytest
from src.core.types import RetrievalResult
from src.core.query_engine.fusion import RRFFusion


class TestRRFFusion:
    def test_fuse_two_lists(self):
        fusion = RRFFusion(k=60)
        dense = [
            RetrievalResult(chunk_id="c1", score=0.9, text="t1"),
            RetrievalResult(chunk_id="c2", score=0.8, text="t2"),
            RetrievalResult(chunk_id="c3", score=0.7, text="t3"),
        ]
        sparse = [
            RetrievalResult(chunk_id="c2", score=2.5, text="t2"),
            RetrievalResult(chunk_id="c4", score=2.0, text="t4"),
            RetrievalResult(chunk_id="c1", score=1.5, text="t1"),
        ]
        results = fusion.fuse(dense, sparse, top_k=3)

        # c1 and c2 appear in both lists, should rank higher
        ids = [r.chunk_id for r in results]
        assert "c1" in ids
        assert "c2" in ids
        assert len(results) == 3

    def test_fuse_deterministic(self):
        fusion = RRFFusion(k=60)
        list1 = [RetrievalResult(chunk_id="a", score=1.0, text="")]
        list2 = [RetrievalResult(chunk_id="b", score=1.0, text="")]
        r1 = fusion.fuse(list1, list2)
        r2 = fusion.fuse(list1, list2)
        assert [x.chunk_id for x in r1] == [x.chunk_id for x in r2]

    def test_fuse_single_list(self):
        fusion = RRFFusion(k=60)
        results = [RetrievalResult(chunk_id="c1", score=0.9, text="t")]
        fused = fusion.fuse(results, top_k=5)
        assert len(fused) == 1
        assert fused[0].chunk_id == "c1"

    def test_fuse_empty_lists(self):
        fusion = RRFFusion(k=60)
        fused = fusion.fuse([], [], top_k=5)
        assert fused == []

    def test_top_k_limits_output(self):
        fusion = RRFFusion(k=60)
        results = [RetrievalResult(chunk_id=f"c{i}", score=float(i), text="") for i in range(20)]
        fused = fusion.fuse(results, top_k=5)
        assert len(fused) == 5

    def test_k_parameter_affects_scores(self):
        # Lower k gives higher scores to top-ranked items
        fusion_low = RRFFusion(k=1)
        fusion_high = RRFFusion(k=100)
        results = [RetrievalResult(chunk_id="c1", score=1.0, text="")]
        low = fusion_low.fuse(results)
        high = fusion_high.fuse(results)
        # With k=1, score = 1/(1+1) = 0.5; with k=100, score = 1/(100+1) ≈ 0.0099
        assert low[0].score > high[0].score
