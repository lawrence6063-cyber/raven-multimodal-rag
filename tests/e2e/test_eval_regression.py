"""E2E regression gate for the IR metrics evaluator (H2-2).

Hermetic (no network, no LLM): builds a small in-memory corpus + a deterministic
keyword search, drives the ``ir`` backend through :class:`EvalRunner` over the
shared golden fixture, and asserts IR metric thresholds. This acts as a CI
regression gate so retrieval or metric regressions fail the build (DeepEval-style
assertion gate).

Corpus chunk ids/sources are kept in sync with
``tests/fixtures/golden_test_set.json``.
"""

from __future__ import annotations

from pathlib import Path

from src.core.settings import Settings
from src.core.types import RetrievalResult
from src.libs.evaluator.evaluator_factory import EvaluatorFactory
from src.observability.evaluation.eval_runner import EvalRunner

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_GOLDEN_SET = _PROJECT_ROOT / "tests" / "fixtures" / "golden_test_set.json"

# Baselines captured from the deterministic keyword search over the fixture.
# The gate allows a small tolerance to avoid flakiness while catching real
# regressions (spec §6.2: ndcg@5 >= baseline - 0.02).
_TOLERANCE = 0.02
_BASELINE = {
    "ir.ndcg@5": 1.0,
    "ir.recall@5": 1.0,
    "ir.hit_rate@5": 1.0,
    "ir.mrr@5": 1.0,
}

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
    {"chunk_id": "doc_misc_001", "source": "misc_notes.md", "text": "这是一段无关的备注信息，用于测试干扰项的影响。"},
    {"chunk_id": "doc_misc_002", "source": "changelog.md", "text": "项目更新日志，记录版本变更与历史发布说明。"},
]


def _bigrams(text: str) -> set[str]:
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


class TestIRRegressionGate:
    """IR metric regression gate over the golden fixture."""

    def _run_report(self):
        settings = Settings()
        settings.evaluation.ks = [1, 3, 5, 10]
        search = _KeywordSearch(_CORPUS)
        evaluator = EvaluatorFactory.create_composite(["ir"])
        runner = EvalRunner(settings, search, evaluator)
        return runner.run(str(_GOLDEN_SET))

    def test_ir_metrics_present(self):
        report = self._run_report()
        for k in (1, 3, 5, 10):
            for name in ("recall", "precision", "mrr", "map", "ndcg", "hit_rate"):
                assert f"ir.{name}@{k}" in report.metrics

    def test_metrics_above_baseline(self):
        report = self._run_report()
        for key, baseline in _BASELINE.items():
            value = report.metrics.get(key, 0.0)
            assert value >= baseline - _TOLERANCE, (
                f"{key}={value:.3f} regressed below baseline {baseline} (tol {_TOLERANCE})"
            )

    def test_latency_metrics_present(self):
        report = self._run_report()
        assert "avg_latency_ms" in report.metrics
        assert "p95_latency_ms" in report.metrics

    def test_per_query_has_ir_breakdown(self):
        report = self._run_report()
        assert report.total_queries == len(report.per_query)
        # EvalRunner merges the ir per-query @k breakdown into each row.
        assert any("ndcg@5" in row for row in report.per_query)
