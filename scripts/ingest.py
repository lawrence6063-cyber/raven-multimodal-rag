#!/usr/bin/env python3
"""Ingestion script — CLI entry point for document ingestion.

Usage:
    # 摄取单个文件（快速测试）
    python scripts/ingest.py --single data/documents/rag/01_rag_knowledge_intensive_nlp.pdf

    # 摄取整个目录
    python scripts/ingest.py --path data/documents

    # 指定 collection + 强制重新处理
    python scripts/ingest.py --path data/documents/rag --collection rag --force

    # 显示详细错误堆栈
    python scripts/ingest.py --single some_file.pdf --verbose
"""

import argparse
import sys
import time
import traceback
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.settings import load_settings, SettingsError
from src.ingestion.pipeline import IngestionPipeline
from src.observability.logger import get_logger

logger = get_logger("ingest")

# 阶段名称中文映射
STAGE_NAMES = {
    "integrity_check": "完整性校验",
    "load": "文档加载",
    "split": "文本分块",
    "transform": "转换增强",
    "encode": "向量编码",
    "store": "持久化存储",
}


def progress_callback(stage: str, current: int, total: int) -> None:
    """进度回调，打印当前阶段进度。"""
    stage_cn = STAGE_NAMES.get(stage, stage)
    if total > 1:
        print(f"    ⏳ {stage_cn} ({current}/{total})...", flush=True)
    else:
        print(f"    ⏳ {stage_cn}...", flush=True)


def ingest_single(settings, file_path: str, collection: str, force: bool, verbose: bool) -> dict:
    """摄取单个文件，带详细进度和耗时输出。"""
    path = Path(file_path)
    if not path.exists():
        print(f"❌ 文件不存在: {file_path}")
        sys.exit(1)
    if not path.is_file():
        print(f"❌ 不是文件: {file_path}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  📄 单文档摄取测试")
    print(f"{'='*60}")
    print(f"  文件: {path.name}")
    print(f"  路径: {path}")
    print(f"  大小: {path.stat().st_size / 1024:.1f} KB")
    print(f"  集合: {collection}")
    print(f"  强制: {'是' if force else '否'}")
    print(f"{'='*60}\n")

    pipeline = IngestionPipeline(settings)

    start_time = time.perf_counter()
    try:
        result = pipeline.run(
            str(path),
            collection=collection,
            force=force,
            on_progress=progress_callback,
        )
        elapsed = time.perf_counter() - start_time

        if result["status"] == "skipped":
            print(f"\n  ⏭️  已跳过（之前已处理过）")
            print(f"  💡 使用 --force 强制重新处理")
        else:
            print(f"\n  ✅ 摄取成功!")
            print(f"  📊 生成 {result['chunk_count']} 个 chunks")
            print(f"  ⏱️  耗时 {elapsed:.2f} 秒")
            if result.get("doc_id"):
                print(f"  🆔 doc_id: {result['doc_id']}")

        print(f"\n{'='*60}\n")
        return result

    except Exception as e:
        elapsed = time.perf_counter() - start_time
        print(f"\n  ❌ 摄取失败!")
        print(f"  ⏱️  耗时 {elapsed:.2f} 秒")
        print(f"  💥 错误: {e}")
        if verbose:
            print(f"\n{'─'*60}")
            traceback.print_exc()
            print(f"{'─'*60}")
        else:
            print(f"  💡 使用 --verbose 查看完整堆栈")
        print(f"\n{'='*60}\n")
        sys.exit(1)


def ingest_batch(settings, path: str, collection: str, force: bool, verbose: bool):
    """批量摄取目录下的所有 PDF。"""
    target = Path(path)
    if target.is_file():
        files = [target]
    elif target.is_dir():
        files = sorted(target.glob("**/*.pdf"))
    else:
        logger.error(f"路径不存在: {path}")
        sys.exit(1)

    if not files:
        logger.warning(f"未找到 PDF 文件: {path}")
        sys.exit(0)

    print(f"\n{'='*60}")
    print(f"  📚 批量文档摄取")
    print(f"{'='*60}")
    print(f"  路径: {target}")
    print(f"  文件数: {len(files)}")
    print(f"  集合: {collection}")
    print(f"  强制: {'是' if force else '否'}")
    print(f"{'='*60}\n")

    pipeline = IngestionPipeline(settings)
    results = []  # (filename, status, chunks, elapsed, error)
    total_start = time.perf_counter()

    for i, file in enumerate(files, 1):
        print(f"  [{i}/{len(files)}] {file.name}")
        file_start = time.perf_counter()
        try:
            result = pipeline.run(str(file), collection=collection, force=force)
            elapsed = time.perf_counter() - file_start
            if result["status"] == "skipped":
                print(f"         ⏭️  跳过 ({elapsed:.1f}s)")
                results.append((file.name, "skipped", 0, elapsed, ""))
            else:
                print(f"         ✅ {result['chunk_count']} chunks ({elapsed:.1f}s)")
                results.append((file.name, "success", result["chunk_count"], elapsed, ""))
        except Exception as e:
            elapsed = time.perf_counter() - file_start
            print(f"         ❌ 失败: {e} ({elapsed:.1f}s)")
            results.append((file.name, "failed", 0, elapsed, str(e)))
            if verbose:
                traceback.print_exc()

    total_elapsed = time.perf_counter() - total_start

    # 汇总
    success = [r for r in results if r[1] == "success"]
    skipped = [r for r in results if r[1] == "skipped"]
    failed = [r for r in results if r[1] == "failed"]
    total_chunks = sum(r[2] for r in results)

    print(f"\n{'='*60}")
    print(f"  📊 摄取汇总")
    print(f"{'─'*60}")
    print(f"  ✅ 成功: {len(success)} 个文件, 共 {total_chunks} chunks")
    print(f"  ⏭️  跳过: {len(skipped)} 个文件")
    print(f"  ❌ 失败: {len(failed)} 个文件")
    print(f"  ⏱️  总耗时: {total_elapsed:.1f} 秒")
    print(f"  📁 集合: {collection}")

    if failed:
        print(f"\n  失败文件:")
        for name, _, _, _, err in failed:
            print(f"    • {name}: {err}")

    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="摄取文档到 RAG 知识库",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单文档测试（推荐先用这个验证流程）
  python scripts/ingest.py --single data/documents/rag/01_rag_knowledge_intensive_nlp.pdf

  # 摄取整个目录
  python scripts/ingest.py --path data/documents

  # 指定 collection
  python scripts/ingest.py --path data/documents/agent --collection agent --force
        """,
    )

    # 互斥组：--single 或 --path
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--single", "-s", help="单文档摄取（快速测试模式）")
    group.add_argument("--path", "-p", help="PDF 文件或目录路径（批量模式）")

    parser.add_argument("--collection", "-c", default="default", help="集合名称 (默认: 'default')")
    parser.add_argument("--force", "-f", action="store_true", help="强制重新处理（忽略去重）")
    parser.add_argument("--verbose", "-v", action="store_true", help="显示详细错误堆栈")
    args = parser.parse_args()

    # 加载配置
    try:
        settings = load_settings()
    except SettingsError as e:
        print(f"❌ 配置错误: {e}")
        sys.exit(1)

    # 执行
    if args.single:
        ingest_single(settings, args.single, args.collection, args.force, args.verbose)
    else:
        ingest_batch(settings, args.path, args.collection, args.force, args.verbose)


if __name__ == "__main__":
    main()
