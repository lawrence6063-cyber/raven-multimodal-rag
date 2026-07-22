"""Fair context selection across agent retrieval requests."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.core.types import RetrievalResult


@dataclass
class RetrievalBatch:
    """Retrieval candidates produced by one sub-query at one hop."""

    request_id: str
    query: str
    hop: int
    results: list[RetrievalResult] = field(default_factory=list)


@dataclass
class ContextSelection:
    """Selected context and auditable request-level allocation metadata."""

    results: list[RetrievalResult] = field(default_factory=list)
    selected_by_request: dict[str, list[str]] = field(default_factory=dict)
    candidate_count: int = 0
    unique_candidate_count: int = 0
    request_count: int = 0

    def to_audit(self) -> dict[str, object]:
        """Return a JSON-friendly selection summary."""
        return {
            "candidate_count": self.candidate_count,
            "unique_candidate_count": self.unique_candidate_count,
            "request_count": self.request_count,
            "selected_count": len(self.results),
            "selected_by_request": self.selected_by_request,
        }


def select_context(batches: list[RetrievalBatch], budget: int) -> ContextSelection:
    """Select de-duplicated context with round-robin request coverage.

    Result order within each request is treated as calibrated, while scores from
    different rewritten queries are not compared directly. Repeated rounds give
    every non-empty request a chance before any request receives another slot.
    """
    candidate_count = sum(len(batch.results) for batch in batches)
    unique_candidate_count = len(
        {result.chunk_id for batch in batches for result in batch.results}
    )
    selected_by_request = {batch.request_id: [] for batch in batches}
    if budget <= 0 or not batches:
        return ContextSelection(
            selected_by_request=selected_by_request,
            candidate_count=candidate_count,
            unique_candidate_count=unique_candidate_count,
            request_count=len(batches),
        )

    best_by_chunk: dict[str, RetrievalResult] = {}
    for batch in batches:
        for result in batch.results:
            current = best_by_chunk.get(result.chunk_id)
            if current is None or result.score > current.score:
                best_by_chunk[result.chunk_id] = result

    positions = [0] * len(batches)
    selected: list[RetrievalResult] = []
    seen: set[str] = set()

    def take_next(index: int) -> bool:
        batch = batches[index]
        while positions[index] < len(batch.results):
            result = batch.results[positions[index]]
            positions[index] += 1
            if result.chunk_id in seen:
                continue
            seen.add(result.chunk_id)
            selected.append(best_by_chunk[result.chunk_id])
            selected_by_request[batch.request_id].append(result.chunk_id)
            return True
        return False

    # Give every request up to two coverage slots, including reflective follow-ups.
    for _ in range(2):
        for index in range(len(batches)):
            take_next(index)
            if len(selected) == budget:
                break
        if len(selected) == budget:
            break

    # Spend remaining budget on earlier hops first. Later reflective requests are
    # exploratory and must not displace the strong context that triggered them.
    for hop in sorted({batch.hop for batch in batches}):
        indices = [index for index, batch in enumerate(batches) if batch.hop == hop]
        while len(selected) < budget:
            progressed = False
            for index in indices:
                progressed = take_next(index) or progressed
                if len(selected) == budget:
                    break
            if not progressed:
                break

    return ContextSelection(
        results=selected,
        selected_by_request=selected_by_request,
        candidate_count=candidate_count,
        unique_candidate_count=unique_candidate_count,
        request_count=len(batches),
    )
