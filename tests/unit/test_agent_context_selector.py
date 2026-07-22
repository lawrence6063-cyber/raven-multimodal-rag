"""Tests for fair cross-request agent context selection."""

from src.core.agent.context_selector import RetrievalBatch, select_context
from src.core.types import RetrievalResult


def _results(prefix: str, count: int, score: float = 1.0) -> list[RetrievalResult]:
    return [
        RetrievalResult(chunk_id=f"{prefix}{index}", score=score - index / 100)
        for index in range(count)
    ]


def test_round_robin_prevents_first_request_monopoly():
    batches = [
        RetrievalBatch("r1", "first", 1, _results("a", 5)),
        RetrievalBatch("r2", "second", 1, _results("b", 5)),
        RetrievalBatch("r3", "third", 2, _results("c", 5)),
    ]

    selection = select_context(batches, budget=6)

    assert [result.chunk_id for result in selection.results] == [
        "a0",
        "b0",
        "c0",
        "a1",
        "b1",
        "c1",
    ]
    assert all(len(ids) == 2 for ids in selection.selected_by_request.values())


def test_deduplicates_and_keeps_best_duplicate_instance():
    batches = [
        RetrievalBatch(
            "r1",
            "first",
            1,
            [
                RetrievalResult(chunk_id="shared", score=0.9),
                RetrievalResult(chunk_id="only-a", score=0.8),
            ],
        ),
        RetrievalBatch(
            "r2",
            "second",
            1,
            [
                RetrievalResult(chunk_id="shared", score=0.4),
                RetrievalResult(chunk_id="only-b", score=0.7),
            ],
        ),
    ]

    selection = select_context(batches, budget=3)

    assert [result.chunk_id for result in selection.results] == [
        "shared",
        "only-b",
        "only-a",
    ]
    assert selection.results[0].score == 0.9
    assert selection.unique_candidate_count == 3


def test_empty_and_zero_budget_are_stable():
    assert select_context([], budget=3).results == []
    selection = select_context(
        [RetrievalBatch("r1", "query", 1, _results("a", 2))], budget=0
    )
    assert selection.results == []
    assert selection.candidate_count == 2
