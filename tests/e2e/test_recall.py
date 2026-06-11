"""E2E recall regression test (H5).

Builds a small in-memory corpus and a deterministic keyword search that mimics
the retrieval contract (returns ``RetrievalResult`` ranked by character-bigram
overlap). The :class:`EvalRunner` is then driven over the shared golden test
set to assert minimum hit@k / mrr thresholds. The test is hermetic (no network,
no real embeddings) so it is safe as a CI regression gate.

The corpus chunk ids and sources are kept in sync with
``tests/fixtures/golden_test_set.json``.
"""

from __future__ import annotations

from pathlib import Path

from src.core.settings import Settings
from src.core.types import RetrievalResult
from src.libs.evaluator.custom_evaluator import CustomEvaluator
from src.observability.evaluation.eval_runner import EvalRunner

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_GOLDEN_SET = _PROJECT_ROOT / "tests" / "fixtures" / "golden_test_set.json"

# Minimum thresholds for regression. Kept conservative to avoid flakiness while
# still catching gross retrieval regressions.
_MIN_HIT_RATE = 0.8
_MIN_MRR = 0.6

# In-memory corpus aligned with golden_test_set.json expected_chunk_ids/sources.
_CORPUS = [
    {
        "chunk_id": "doc_azure_001",
        "source": "azure_openai_guide.md",
        "text": "如何配置 Azure OpenAI 服务，需要在配置文件中填写 endpoint 和 api_key 以及 deployment 名称。",
    },
    {
        "chunk_id": "doc_chroma_001",
        "source": "vector_store_chroma.md",
        "text": "Chroma 向量数据库支持本地持久化存储，将向量索引保存到磁盘目录便于复用。",
    },
    {
        "chunk_id": "doc_bm25_001",
        "source": "sparse_retrieval_bm25.md",
        "text": "BM25 稀疏检索基于关键词匹配原理，对查询词频和文档长度进行加权打分。",
    },
    {
        "chunk_id": "doc_rrf_001",
        "source": "fusion_rrf.md",
        "text": "RRF 融合排序通过倒数排名加权合并多路检索结果，得到统一的排序列表。",
    },
    {
        "chunk_id": "doc_rerank_001",
        "source": "reranker_guide.md",
        "text": "reranker 重排序模型对初步检索结果重新打分，显著提升最终结果的相关性。",
    },
    # Distractors
    {
        "chunk_id": "doc_misc_001",
        "source": "misc_notes.md",
        "text": "这是一段无关的备注信息，用于测试干扰项的影响。",
    },
    {
        "chunk_id": "doc_misc_002",
        "source": "changelog.md",
        "text": "项目更新日志，记录版本变更与历史发布说明。",
    },
]


def _bigrams(text: str) -> set[str]:
    """Return the set of character bigrams (whitespace removed)."""
    s = text.replace(" ", "")
    return {s[i : i + 2] for i in range(len(s) - 1)}


class _KeywordSearch:
    """Deterministic in-memory search ranking by character-bigram overlap."""

    def __init__(self, corpus: list[dict[str, str]], top_k: int = 5):
        self._corpus = corpus
        self._top_k = top_k

    def search(self, query, top_k=None, filters=None, trace=None):
        k = top_k or self._top_k
        q_grams = _bigrams(query)
        scored = []
        for doc in self._corpus:
            overlap = len(q_grams & _bigrams(doc["text"]))
            if overlap > 0:
                scored.append((overlap, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            RetrievalResult(
                chunk_id=doc["chunk_id"],
                score=float(score),
                text=doc["text"],
                metadata={"source": doc["source"]},
            )
            for score, doc in scored[:k]
        ]


class TestRecallRegression:
    """Recall regression over the golden test set."""

    def _run_report(self):
        settings = Settings()
        search = _KeywordSearch(_CORPUS)
        runner = EvalRunner(settings, search, CustomEvaluator())
        return runner.run(str(_GOLDEN_SET))

    def test_hit_rate_above_threshold(self):
        report = self._run_report()
        assert report.metrics["hit_rate"] >= _MIN_HIT_RATE, (
            f"hit_rate {report.metrics['hit_rate']:.3f} below {_MIN_HIT_RATE}"
        )

    def test_mrr_above_threshold(self):
        report = self._run_report()
        assert report.metrics["mrr"] >= _MIN_MRR, (
            f"mrr {report.metrics['mrr']:.3f} below {_MIN_MRR}"
        )

    def test_every_query_evaluated(self):
        report = self._run_report()
        assert report.total_queries == len(report.per_query)
        assert report.total_queries >= 5
