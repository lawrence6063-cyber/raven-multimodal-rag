#!/usr/bin/env python3
"""gen_testset.py — synthesize a chunk-level graded golden test set.

Two complementary strategies (spec §3.2), both emitting the §3.1 format
(``expected_chunk_ids`` + graded ``relevance`` + optional ``answer``):

* ``llamaindex`` (chunk-level native): enumerate chunks already ingested in the
  vector store, ask the project LLM to write 1-2 questions per chunk. The source
  chunk *is* the golden node, so ``expected_chunk_ids`` are exact and
  ``relevance`` defaults to grade 3. Requires an ingested DB + an LLM key.
* ``ragas``: use ``ragas.testset.TestsetGenerator`` over ``data/documents/`` to
  synthesize QA with ground-truth answers. Requires the ``ragas`` package + key.

Security: LLM/embedding keys are read only from the environment (via the project
settings' ``_resolve_api_keys`` fallback). No key -> the script prints a clear
error and exits non-zero WITHOUT writing a partial file.

Usage:
    python scripts/gen_testset.py --strategy llamaindex --n 150 --out data/golden_synth_li.json
    python scripts/gen_testset.py --strategy ragas      --n 150 --out data/golden_synth_ragas.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.settings import SettingsError, load_settings
from src.observability.logger import get_logger

logger = get_logger("gen_testset")

# _QUESTION_PROMPT instructs the LLM to produce standalone questions per chunk
_QUESTION_PROMPT = (
    "你是一个检索评测数据集构造助手。基于下面的文档片段，写 {n} 个"
    "**可以仅凭该片段回答**的、具体的中文或英文问题（与片段语言一致）。"
    "要求：问题自包含、不含指代词、不要提到\"这段/上文\"等词。"
    "只输出问题，每行一个，不要编号、不要多余说明。\n\n片段：\n{chunk}"
)


def _require_key(settings) -> None:
    """Abort early with a clear message when no LLM key is configured."""
    if not settings.llm.api_key:
        print(
            "\n❌ 未检测到 LLM API Key。请设置环境变量后重试："
            "\n   export DASHSCOPE_API_KEY=...   # 或 OPENAI_API_KEY=..."
            "\n（key 仅从环境变量读取，禁止写入 settings.yaml）",
            file=sys.stderr,
        )
        sys.exit(2)


def _parse_questions(raw: str) -> list[str]:
    """Split an LLM reply into individual question strings."""
    lines = [ln.strip() for ln in raw.splitlines()]
    out = []
    for ln in lines:
        if not ln:
            continue
        # strip leading numbering like "1." / "1)" / "- "
        ln = re.sub(r"^\s*(?:\d+[.)]|[-*])\s*", "", ln)
        if len(ln) >= 5:
            out.append(ln)
    return out


def _round_robin_by_source(chunks: list, skip_head: int = 2) -> list:
    """Interleave chunks by their source document for balanced coverage.

    Groups chunks by ``source_path``/``file_name``, sorts each group by
    ``chunk_index`` and drops the first ``skip_head`` chunks (title / copyright /
    abstract front-matter pages that yield low-value meta questions). Then emits
    one chunk per source per round so a capped ``--n`` spreads across all
    documents. A document with too few chunks keeps all of them (never emptied).
    """
    groups: dict[str, list] = {}
    for rec in chunks:
        src = rec.metadata.get("source_path") or rec.metadata.get("file_name") or "unknown"
        groups.setdefault(src, []).append(rec)
    trimmed: dict[str, list] = {}
    for src, lst in groups.items():
        lst.sort(key=lambda r: r.metadata.get("chunk_index", 0))
        trimmed[src] = lst[skip_head:] if len(lst) > skip_head else lst
    ordered: list = []
    i = 0
    while True:
        added = False
        for lst in trimmed.values():
            if i < len(lst):
                ordered.append(lst[i])
                added = True
        if not added:
            break
        i += 1
    return ordered


def _gen_llamaindex(settings, n: int, per_chunk: int) -> list[dict[str, Any]]:
    """Chunk-level native strategy: enumerate store chunks -> LLM questions."""
    from src.libs.llm.llm_factory import LLMFactory
    from src.libs.llm.base_llm import ChatMessage
    from src.libs.vector_store.vector_store_factory import VectorStoreFactory

    store = VectorStoreFactory.create(settings.vector_store)
    get_all = getattr(store, "get_all", None)
    if get_all is None:
        print("\n❌ 当前向量库不支持枚举全部 chunk（缺 get_all）。", file=sys.stderr)
        sys.exit(1)
    chunks = get_all()
    if not chunks:
        print(
            "\n❌ 向量库为空。请先摄取文档：python scripts/ingest.py --path data/documents",
            file=sys.stderr,
        )
        sys.exit(1)

    # 跨文档 round-robin 采样，保证 n 条用例均匀覆盖所有来源文档，
    # 而非集中在存储顺序靠前的少数几篇。
    ordered = _round_robin_by_source(chunks)

    llm = LLMFactory.create(settings.llm)
    cases: list[dict[str, Any]] = []
    for rec in ordered:
        if len(cases) >= n:
            break
        text = (rec.text or "").strip()
        if len(text) < 80:  # skip trivially short chunks
            continue
        source = rec.metadata.get("source_path") or rec.metadata.get("file_name") or ""
        prompt = _QUESTION_PROMPT.format(n=per_chunk, chunk=text[:1500])
        try:
            reply = llm.chat([ChatMessage(role="user", content=prompt)])
        except Exception as exc:  # noqa: BLE001 - one bad chunk must not abort
            logger.warning(f"LLM question gen failed for {rec.id}: {exc}")
            continue
        for question in _parse_questions(reply.content)[:per_chunk]:
            cases.append(
                {
                    "query": question,
                    "expected_chunk_ids": [rec.id],
                    "expected_sources": [source] if source else [],
                    "relevance": {rec.id: 3},
                    "category": "semantic",
                    "source": "llamaindex_synth",
                }
            )
            if len(cases) >= n:
                break
    return cases


def _gen_ragas(settings, n: int, docs_dir: str) -> list[dict[str, Any]]:
    """Ragas strategy: TestsetGenerator over documents with ground-truth answers."""
    try:
        import ragas  # noqa: F401
    except ImportError:
        print(
            "\n❌ 未安装 ragas。请安装：pip install ragas"
            "\n或改用 --strategy llamaindex（复用项目 LLM，无需 ragas）。",
            file=sys.stderr,
        )
        sys.exit(1)

    # Ragas' generator API varies across versions; we degrade gracefully and
    # surface a clear message rather than guessing an incompatible signature.
    print(
        "\nℹ️ ragas 路线需要与已安装的 ragas 版本对齐 TestsetGenerator API。"
        "\n   请参考 docs/EVAL_UPGRADE_SPEC.md §3.2，将生成结果转为 §3.1 格式后"
        "\n   用 scripts/merge_testset.py 并入 golden_papers.json。"
        f"\n   （目标条数 n={n}，文档目录={docs_dir}）",
        file=sys.stderr,
    )
    # Best-effort: return empty so the caller does not write a bogus file.
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a chunk-level graded golden test set")
    parser.add_argument(
        "--strategy",
        required=True,
        choices=["llamaindex", "ragas"],
        help="llamaindex(chunk-level native) | ragas(QA synthesis)",
    )
    parser.add_argument("--n", type=int, default=150, help="Target number of cases")
    parser.add_argument("--per-chunk", type=int, default=1, help="Questions per chunk (llamaindex)")
    parser.add_argument("--docs", default="data/documents", help="Documents dir (ragas)")
    parser.add_argument("--out", required=True, help="Output JSON path")
    args = parser.parse_args()

    try:
        settings = load_settings()
    except SettingsError as exc:
        print(f"\n❌ 配置错误: {exc}", file=sys.stderr)
        sys.exit(1)

    _require_key(settings)

    if args.strategy == "llamaindex":
        cases = _gen_llamaindex(settings, args.n, args.per_chunk)
    else:
        cases = _gen_ragas(settings, args.n, args.docs)

    if not cases:
        print("\n❌ 未生成任何用例，不写出文件。", file=sys.stderr)
        sys.exit(1)

    out = {
        "description": f"Synthetic golden test set ({args.strategy}), {len(cases)} cases.",
        "test_cases": cases,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✅ 生成 {len(cases)} 条用例 -> {out_path}", file=sys.stderr)
    print("💡 建议人工抽检后用 scripts/merge_testset.py 并入 golden_papers.json", file=sys.stderr)


if __name__ == "__main__":
    main()
