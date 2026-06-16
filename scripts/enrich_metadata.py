#!/usr/bin/env python3
"""Offline metadata enrichment — batch LLM enrich after ingest (方案B).

Reads chunks with enriched_by=rule from Chroma, calls LLM concurrently to
generate title/summary/tags, then updates metadata back to Chroma.

Usage:
    # 对指定 collection 补 LLM metadata
    python scripts/enrich_metadata.py --collection agent

    # 对所有未 LLM enrich 的 chunks 补 metadata
    python scripts/enrich_metadata.py

    # 控制并发数（默认 5，DashScope QPS 约 10-20）
    python scripts/enrich_metadata.py --collection rag --concurrency 10

    # 仅处理前 N 个 chunks（测试用）
    python scripts/enrich_metadata.py --collection agent --limit 10

    # dry-run 模式（不回写，仅打印）
    python scripts/enrich_metadata.py --collection agent --dry-run
"""

import argparse
import json
import logging
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.settings import load_settings, SettingsError
from src.libs.llm.llm_factory import LLMFactory
from src.libs.llm.base_llm import ChatMessage
from src.libs.vector_store.vector_store_factory import VectorStoreFactory
from src.observability.logger import get_logger

logger = get_logger("enrich_metadata")

# Prompt 加载
_PROMPT_PATH = Path("config/prompts/metadata_enrichment.txt")
_PROMPT_DELIMITER = "---USER---"
_MAX_TEXT_LEN = 1000


def _load_prompt() -> tuple[str, str]:
    """Load prompt template from file."""
    if _PROMPT_PATH.exists():
        content = _PROMPT_PATH.read_text(encoding="utf-8")
        if _PROMPT_DELIMITER in content:
            parts = content.split(_PROMPT_DELIMITER, 1)
            return parts[0].strip(), parts[1].strip()
        return "", content.strip()
    return (
        "你是一个文档元数据提取助手。",
        "请为以下文档片段提取 title、summary、tags，返回 JSON。\n\n{text}",
    )


def _parse_json(text: str | None) -> dict | None:
    """Robustly extract JSON from LLM response."""
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


def _enrich_one(llm, system_prompt: str, user_template: str, chunk_id: str, text: str) -> tuple[str, dict | None, float]:
    """Enrich a single chunk. Returns (chunk_id, metadata_dict_or_None, elapsed_seconds)."""
    t0 = time.perf_counter()
    try:
        user_content = user_template.replace("{text}", text[:_MAX_TEXT_LEN])
        messages = []
        if system_prompt:
            messages.append(ChatMessage(role="system", content=system_prompt))
        messages.append(ChatMessage(role="user", content=user_content))

        response = llm.chat(messages)
        data = _parse_json(response.content)
        elapsed = time.perf_counter() - t0

        if data is None:
            logger.warning("  [%s] JSON parse failed (%.2fs): %r", chunk_id[:12], elapsed, (response.content or "")[:100])
            return (chunk_id, None, elapsed)

        meta = {
            "title": str(data.get("title", "")).strip(),
            "summary": str(data.get("summary", "")).strip(),
            "tags": data.get("tags", []) if isinstance(data.get("tags"), list) else [],
            "enriched_by": "llm",
        }
        return (chunk_id, meta, elapsed)
    except Exception as e:
        elapsed = time.perf_counter() - t0
        logger.error("  [%s] LLM call failed (%.2fs): %s", chunk_id[:12], elapsed, e)
        return (chunk_id, None, elapsed)


def main():
    parser = argparse.ArgumentParser(
        description="离线批量 LLM metadata enrichment（方案B）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--collection", "-c", default=None, help="指定 collection（默认全部）")
    parser.add_argument("--concurrency", "-j", type=int, default=5, help="并发线程数 (默认: 5)")
    parser.add_argument("--limit", "-l", type=int, default=None, help="最多处理 N 个 chunks（测试用）")
    parser.add_argument("--dry-run", action="store_true", help="仅打印不回写")
    parser.add_argument("--batch-size", "-b", type=int, default=50, help="每批回写大小 (默认: 50)")
    args = parser.parse_args()

    # 加载配置
    try:
        settings = load_settings()
    except SettingsError as e:
        print(f"❌ 配置错误: {e}")
        sys.exit(1)

    # 初始化
    chroma = VectorStoreFactory.create(settings.vector_store)
    llm = LLMFactory.create(settings.llm)
    system_prompt, user_template = _load_prompt()

    # 查找需要 enrich 的 chunks
    print(f"\n{'='*60}")
    print(f"  🔄 离线 LLM Metadata Enrichment")
    print(f"{'='*60}")

    where_filter: dict = {"enriched_by": "rule"}
    if args.collection:
        # Chroma where 只支持单 key 或 $and/$or
        where_filter = {"$and": [{"enriched_by": "rule"}, {"collection": args.collection}]}

    print(f"  查询条件: {where_filter}")
    print(f"  并发数: {args.concurrency}")
    print(f"  Dry-run: {'是' if args.dry_run else '否'}")

    try:
        chunks = chroma.get_all_by_metadata(where=where_filter, limit=args.limit or 10000)
    except Exception as e:
        print(f"❌ 查询 Chroma 失败: {e}")
        sys.exit(1)

    if not chunks:
        print(f"\n  ✅ 没有需要 enrich 的 chunks（所有 chunks 已是 enriched_by=llm）")
        print(f"{'='*60}\n")
        return

    total = len(chunks)
    if args.limit and total > args.limit:
        chunks = chunks[:args.limit]
        total = args.limit

    print(f"  待处理: {total} chunks")
    print(f"  模型: {settings.llm.model}")
    print(f"{'='*60}\n")

    # 并发 enrich
    start_time = time.perf_counter()
    results: list[tuple[str, dict | None, float]] = []
    completed_count = 0

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(_enrich_one, llm, system_prompt, user_template, chunk.id, chunk.text or ""): chunk
            for chunk in chunks
        }

        for future in as_completed(futures):
            chunk_id, meta, elapsed = future.result()
            results.append((chunk_id, meta, elapsed))
            completed_count += 1

            if meta:
                print(f"  ✅ [{completed_count}/{total}] ({elapsed:.1f}s) title={meta.get('title', '')[:50]}")
            else:
                print(f"  ⚠️  [{completed_count}/{total}] ({elapsed:.1f}s) failed, skipped")

    total_time = time.perf_counter() - start_time

    # 统计
    success_results = [(cid, m, t) for cid, m, t in results if m is not None]
    failed_count = total - len(success_results)

    print(f"\n{'─'*60}")
    print(f"  LLM 调用完成: {len(success_results)} 成功, {failed_count} 失败")
    print(f"  总耗时: {total_time:.1f}s (avg {total_time/max(total,1):.2f}s/chunk)")
    print(f"{'─'*60}")

    # 批量回写
    if args.dry_run:
        print(f"\n  🏷️  Dry-run 模式，跳过回写")
    elif success_results:
        print(f"\n  📝 回写 metadata 到 Chroma...")
        written = 0
        batch_ids: list[str] = []
        batch_metas: list[dict] = []

        for chunk_id, meta, _ in success_results:
            # 合并原有 metadata + 新 LLM metadata
            original_chunk = next((c for c in chunks if c.id == chunk_id), None)
            if original_chunk:
                merged = dict(original_chunk.metadata or {})
                merged.update(meta)
            else:
                merged = meta

            batch_ids.append(chunk_id)
            batch_metas.append(merged)

            if len(batch_ids) >= args.batch_size:
                chroma.update_metadata(batch_ids, batch_metas)
                written += len(batch_ids)
                print(f"    写入 {written}/{len(success_results)}...")
                batch_ids = []
                batch_metas = []

        # 剩余
        if batch_ids:
            chroma.update_metadata(batch_ids, batch_metas)
            written += len(batch_ids)

        print(f"  ✅ 回写完成: {written} chunks 已更新")

    # 汇总
    print(f"\n{'='*60}")
    print(f"  📊 Enrichment 汇总")
    print(f"{'─'*60}")
    print(f"  处理: {total} chunks")
    print(f"  成功: {len(success_results)}")
    print(f"  失败: {failed_count}")
    print(f"  耗时: {total_time:.1f}s")
    if args.collection:
        print(f"  集合: {args.collection}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
