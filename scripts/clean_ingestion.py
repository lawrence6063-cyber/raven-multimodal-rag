#!/usr/bin/env python3
"""清理 Ingestion 数据脚本 — 删除所有或部分摄取产生的持久化数据，以便重新 ingest。

Usage:
    python scripts/clean_ingestion.py                    # 交互确认后清理全部
    python scripts/clean_ingestion.py --yes              # 跳过确认直接清理全部
    python scripts/clean_ingestion.py --only chroma bm25 # 只清理指定组件
    python scripts/clean_ingestion.py --except images    # 清理除图片外的所有
    python scripts/clean_ingestion.py --logs             # 同时清理日志
    python scripts/clean_ingestion.py --dry-run          # 仅预览，不实际删除
"""

import argparse
import shutil
import sys
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# ============================================================
# 可清理的组件定义（key 用于 CLI 参数选择）
# ============================================================
COMPONENTS = {
    "chroma": {
        "name": "Chroma 向量库",
        "path": "data/db/chroma",
        "description": "Dense 向量检索数据（ChromaDB 持久化目录）",
    },
    "bm25": {
        "name": "BM25 倒排索引",
        "path": "data/db/bm25",
        "description": "Sparse 检索的倒排索引（pickle 文件）",
    },
    "image_db": {
        "name": "图片索引数据库",
        "path": "data/db/image_index.db",
        "description": "图片 ID → 文件路径的 SQLite 映射",
    },
    "history": {
        "name": "摄取历史数据库",
        "path": "data/db/ingestion_history.db",
        "description": "文件去重记录（SHA256 → 处理状态）",
    },
    "images": {
        "name": "提取的图片文件",
        "path": "data/images",
        "description": "从 PDF 中提取的图片文件",
    },
    "logs": {
        "name": "日志文件",
        "path": "logs",
        "description": "Trace 日志和运行日志",
    },
}

# 默认清理的组件（不含 logs）
DEFAULT_COMPONENTS = ["chroma", "bm25", "image_db", "history", "images"]


def get_size(path: Path) -> str:
    """获取文件/目录大小的可读字符串。"""
    if not path.exists():
        return "不存在"
    if path.is_file():
        size = path.stat().st_size
    else:
        size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())

    if size < 1024:
        return f"{size} B"
    elif size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    elif size < 1024 * 1024 * 1024:
        return f"{size / (1024 * 1024):.1f} MB"
    else:
        return f"{size / (1024 * 1024 * 1024):.2f} GB"


def remove_path(path: Path) -> bool:
    """安全删除文件或目录，返回是否成功。"""
    if not path.exists():
        return True
    try:
        if path.is_file():
            path.unlink()
        else:
            shutil.rmtree(path)
        return True
    except Exception as e:
        print(f"  ✗ 删除失败: {path} — {e}")
        return False


def resolve_components(args) -> list[str]:
    """根据 CLI 参数解析最终要清理的组件列表。"""
    if args.only:
        # 验证用户指定的组件名
        for key in args.only:
            if key not in COMPONENTS:
                print(f"❌ 未知组件: '{key}'")
                print(f"   可选: {', '.join(COMPONENTS.keys())}")
                sys.exit(1)
        return args.only

    # 基础列表
    selected = list(DEFAULT_COMPONENTS)

    # --logs 追加日志
    if args.logs:
        selected.append("logs")

    # --all 包含全部
    if args.all:
        selected = list(COMPONENTS.keys())

    # --except 排除
    if getattr(args, "except_", None):
        for key in args.except_:
            if key not in COMPONENTS:
                print(f"❌ 未知组件: '{key}'")
                print(f"   可选: {', '.join(COMPONENTS.keys())}")
                sys.exit(1)
            if key in selected:
                selected.remove(key)

    return selected


def main():
    parser = argparse.ArgumentParser(
        description="清理 Ingestion 产生的数据（支持细粒度选择）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
可清理的组件:
  {"组件 Key":<12} {"名称":<16} 说明
  {"─"*12} {"─"*16} {"─"*40}
"""
        + "\n".join(
            f"  {key:<12} {comp['name']:<16} {comp['description']}"
            for key, comp in COMPONENTS.items()
        )
        + """

示例:
  # 清理全部（默认不含 logs）
  python scripts/clean_ingestion.py --yes

  # 只清理向量库和 BM25（保留历史记录，下次 ingest 会跳过已处理文件）
  python scripts/clean_ingestion.py --only chroma bm25

  # 清理除图片外的所有
  python scripts/clean_ingestion.py --except images

  # 全部清理（含日志）
  python scripts/clean_ingestion.py --all --yes

  # 只清理历史记录（让 ingest 重新处理所有文件）
  python scripts/clean_ingestion.py --only history
""",
    )

    parser.add_argument("--yes", "-y", action="store_true", help="跳过交互确认，直接清理")
    parser.add_argument("--dry-run", action="store_true", help="仅显示将要删除的内容，不实际执行")
    parser.add_argument("--logs", action="store_true", help="同时清理日志文件")
    parser.add_argument("--all", action="store_true", help="清理所有组件（含日志）")
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="COMPONENT",
        help=f"只清理指定组件 (可选: {', '.join(COMPONENTS.keys())})",
    )
    parser.add_argument(
        "--except",
        nargs="+",
        dest="except_",
        metavar="COMPONENT",
        help=f"排除指定组件不清理 (可选: {', '.join(COMPONENTS.keys())})",
    )
    parser.add_argument("--list", action="store_true", help="列出所有可清理组件及当前状态")
    args = parser.parse_args()

    # --list 模式：仅展示状态
    if args.list:
        print(f"\n{'='*60}")
        print("  Modular RAG — 数据组件状态")
        print(f"{'='*60}\n")
        for key, comp in COMPONENTS.items():
            full_path = PROJECT_ROOT / comp["path"]
            size = get_size(full_path)
            exists = full_path.exists()
            status = f"✅ {size}" if exists else "⬜ 不存在"
            print(f"  {key:<12} {comp['name']:<16} {status}")
        print()
        return

    # 解析要清理的组件
    selected = resolve_components(args)

    if not selected:
        print("⚠️  没有选中任何组件，无需清理。")
        return

    # 显示将要清理的内容
    print(f"\n{'='*60}")
    print("  Modular RAG — Ingestion 数据清理工具")
    print(f"{'='*60}\n")
    print("将要清理以下组件：\n")

    existing_targets = {}
    for key in selected:
        comp = COMPONENTS[key]
        full_path = PROJECT_ROOT / comp["path"]
        size = get_size(full_path)
        exists = full_path.exists()
        status = f"({size})" if exists else "(不存在，跳过)"
        marker = "🗑️ " if exists else "⏭️ "
        print(f"  {marker} [{key}] {comp['name']:<16} {comp['path']:<30} {status}")
        if exists:
            existing_targets[key] = full_path

    # 显示未选中的组件
    not_selected = [k for k in COMPONENTS if k not in selected]
    if not_selected:
        print(f"\n  保留不清理：")
        for key in not_selected:
            comp = COMPONENTS[key]
            full_path = PROJECT_ROOT / comp["path"]
            if full_path.exists():
                print(f"  ⏸️  [{key}] {comp['name']}")

    print()

    if not existing_targets:
        print("✅ 没有需要清理的数据，环境已经是干净的。")
        return

    if args.dry_run:
        print("(--dry-run 模式，未实际删除任何内容)")
        return

    # 确认
    if not args.yes:
        print("⚠️  此操作不可逆！删除后需要重新运行 ingest 才能恢复数据。")
        answer = input("\n确认清理？[y/N]: ").strip().lower()
        if answer not in ("y", "yes"):
            print("已取消。")
            sys.exit(0)

    # 执行清理
    print()
    success_count = 0
    fail_count = 0

    for key, full_path in existing_targets.items():
        comp = COMPONENTS[key]
        if remove_path(full_path):
            print(f"  ✓ 已删除: {comp['name']}")
            success_count += 1
        else:
            fail_count += 1

    # 重建必要的空目录（保持 git 结构）
    dirs_to_recreate = []
    if "chroma" in existing_targets or "bm25" in existing_targets or "image_db" in existing_targets or "history" in existing_targets:
        dirs_to_recreate.append(PROJECT_ROOT / "data" / "db")
    if "images" in existing_targets:
        dirs_to_recreate.append(PROJECT_ROOT / "data" / "images")
    if "logs" in existing_targets:
        dirs_to_recreate.append(PROJECT_ROOT / "logs")

    for d in dirs_to_recreate:
        d.mkdir(parents=True, exist_ok=True)
        gitkeep = d / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.touch()

    print()
    print(f"{'='*60}")
    print(f"  清理完成: {success_count} 项成功", end="")
    if fail_count:
        print(f", {fail_count} 项失败")
    else:
        print()
    print()
    print("  现在可以重新运行 ingestion：")
    print("    .venv/bin/python scripts/ingest.py -s <file.pdf>       # 单文档测试")
    print("    .venv/bin/python scripts/ingest.py -p data/documents   # 批量摄取")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
