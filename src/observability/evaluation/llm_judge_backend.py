"""llm_judge_backend.py — generation-side metrics via an LLM judge (no ragas dep).

The ``ragas`` package pulls in a fragile langchain stack and defaults to OpenAI
for both the judge LLM and embeddings. This module provides a drop-in
``RagasBackend`` callable that computes the same three generation metrics using
the project's already-integrated LLM (DeepSeek) and embedding (DashScope)
providers, so ``--backends ragas`` works with the keys we actually have.

Metric definitions (aligned with Ragas):
- faithfulness: fraction of answer claims supported by the retrieved context.
- answer_relevancy: mean cosine similarity between the original question and
  questions an LLM reconstructs from the answer.
- context_precision: rank-weighted precision of the retrieved contexts judged
  relevant to the question.

Every LLM/embedding call is defensive: a per-sample failure degrades that
sample's metric to 0.0 and is logged, never aborting the batch.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from src.core.agent.json_utils import extract_json_object
from src.libs.evaluator.base_evaluator import EvalInput
from src.libs.llm.base_llm import ChatMessage
from src.observability.logger import get_logger

if TYPE_CHECKING:
    from src.core.settings import Settings

logger = get_logger("evaluation.llm_judge")

_FAITHFULNESS_PROMPT = (
    "You are evaluating whether an ANSWER is faithful to the given CONTEXT.\n"
    "Extract the distinct factual claims made in the ANSWER, then count how many "
    "are directly supported by the CONTEXT.\n\n"
    "QUESTION: {question}\n\nANSWER: {answer}\n\nCONTEXT:\n{context}\n\n"
    'Output ONLY JSON: {{"total": <int>, "supported": <int>}}'
)

_QUESTIONS_PROMPT = (
    "Generate 3 diverse questions that the following ANSWER would directly and "
    "fully answer. Keep each question self-contained.\n\n"
    "ANSWER: {answer}\n\n"
    'Output ONLY JSON: {{"questions": ["...", "...", "..."]}}'
)

_CONTEXT_PRECISION_PROMPT = (
    "Given a QUESTION and numbered CONTEXT passages, decide for EACH passage "
    "whether it is useful for answering the question.\n\n"
    "QUESTION: {question}\n\nCONTEXT PASSAGES:\n{context}\n\n"
    'Output ONLY JSON: {{"relevances": [<0 or 1 for each passage, in order>]}}'
)


def build_llm_judge_backend(settings: "Settings"):
    """Build a RagasBackend callable backed by the project's LLM + embedding.

    Args:
        settings: Root settings providing the ``llm`` and ``embedding`` configs.

    Returns:
        A callable ``(samples, metric_names) -> dict[str, float]`` averaging each
        requested metric over the samples.
    """
    from src.libs.embedding.embedding_factory import EmbeddingFactory
    from src.libs.llm.llm_factory import LLMFactory

    llm = LLMFactory.create(settings.llm)
    embedding = EmbeddingFactory.create(settings.embedding)

    def _chat_json(prompt: str) -> dict | None:
        try:
            resp = llm.chat([ChatMessage(role="user", content=prompt)])
            return extract_json_object(resp.content)
        except Exception as exc:  # noqa: BLE001 - degrade this metric to 0
            logger.warning(f"judge LLM call failed: {exc}")
            return None

    def _faithfulness(sample: EvalInput) -> float:
        if not sample.answer.strip():
            return 0.0
        context = "\n".join(sample.contexts or sample.retrieved_texts)
        data = _chat_json(
            _FAITHFULNESS_PROMPT.format(
                question=sample.query, answer=sample.answer, context=context
            )
        )
        if not data:
            return 0.0
        total = _as_int(data.get("total"))
        supported = _as_int(data.get("supported"))
        if total <= 0:
            return 0.0
        return max(0.0, min(1.0, supported / total))

    def _answer_relevancy(sample: EvalInput) -> float:
        if not sample.answer.strip():
            return 0.0
        data = _chat_json(_QUESTIONS_PROMPT.format(answer=sample.answer))
        gen = [q for q in (data or {}).get("questions", []) if isinstance(q, str) and q.strip()]
        if not gen:
            return 0.0
        try:
            vectors = embedding.embed([sample.query] + gen)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"judge embedding failed: {exc}")
            return 0.0
        q_vec = vectors[0]
        sims = [_cosine(q_vec, v) for v in vectors[1:] if v]
        return sum(sims) / len(sims) if sims else 0.0

    def _context_precision(sample: EvalInput) -> float:
        contexts = sample.contexts or sample.retrieved_texts
        if not contexts:
            return 0.0
        numbered = "\n".join(f"[{i + 1}] {c[:600]}" for i, c in enumerate(contexts))
        data = _chat_json(
            _CONTEXT_PRECISION_PROMPT.format(question=sample.query, context=numbered)
        )
        rels = (data or {}).get("relevances", [])
        flags = [1 if _as_int(r) > 0 else 0 for r in rels][: len(contexts)]
        if not flags or sum(flags) == 0:
            return 0.0
        # Ragas-style rank-weighted precision: Σ (precision@k · v_k) / Σ v_k.
        hits = 0
        weighted = 0.0
        for k, v in enumerate(flags, start=1):
            if v:
                hits += 1
                weighted += hits / k
        return weighted / sum(flags)

    _COMPUTERS = {
        "faithfulness": _faithfulness,
        "answer_relevancy": _answer_relevancy,
        "context_precision": _context_precision,
    }

    def backend(samples: list[EvalInput], metric_names: tuple[str, ...]) -> dict[str, float]:
        results: dict[str, float] = {}
        for name in metric_names:
            fn = _COMPUTERS.get(name)
            if fn is None:
                results[name] = 0.0
                continue
            scores = [fn(s) for s in samples]
            results[name] = sum(scores) / len(scores) if scores else 0.0
        return results

    return backend


def _as_int(value) -> int:
    """Best-effort int coercion returning 0 on failure."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two vectors (0.0 when either is degenerate)."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0
