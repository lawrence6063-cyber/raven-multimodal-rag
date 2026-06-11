"""Ingestion Manager page — upload, ingest, progress, and delete (G4)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from src.observability.dashboard.services.data_service import DataService


def render():
    """Render the ingestion manager page."""
    st.title("📥 Ingestion 管理")
    st.markdown("上传文件触发摄取、查看进度、管理已有文档。")

    data_service = DataService()

    # --- Upload Section ---
    st.subheader("📤 上传文件")

    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_file = st.file_uploader(
            "选择要摄取的文件",
            type=["pdf", "txt", "md", "docx"],
            help="支持 PDF、TXT、Markdown、DOCX 格式",
        )
    with col2:
        collection = st.text_input("集合名称", value="default")
        force = st.checkbox("强制重新处理", value=False)

    if uploaded_file and st.button("🚀 开始摄取", type="primary"):
        _run_ingestion(uploaded_file, collection, force)

    # --- Document List Section ---
    st.markdown("---")
    st.subheader("📋 已摄入文档")

    try:
        docs = data_service.list_documents()
    except Exception as e:
        st.error(f"加载文档列表失败：{e}")
        return

    if not docs:
        st.info("暂无已摄入的文档。")
        return

    for doc in docs:
        col1, col2, col3 = st.columns([4, 1, 1])
        with col1:
            st.markdown(
                f"**{Path(doc.source_path).name}** "
                f"({doc.collection}) — {doc.chunk_count} chunks, {doc.image_count} 图片"
            )
        with col2:
            st.caption(doc.processed_at[:10] if doc.processed_at else "")
        with col3:
            if st.button("🗑️ 删除", key=f"del_{doc.file_hash}"):
                _delete_document(data_service, doc.source_path, doc.collection)


def _run_ingestion(uploaded_file, collection: str, force: bool):
    """Execute ingestion pipeline with progress display."""
    # Save uploaded file to temp location
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    progress_bar = st.progress(0, text="准备中...")
    status_text = st.empty()

    stage_progress = {"integrity_check": 5, "load": 15, "split": 30, "transform": 60, "encode": 80, "store": 95}

    def on_progress(stage_name: str, current: int, total: int):
        pct = stage_progress.get(stage_name, 50)
        progress_bar.progress(pct / 100, text=f"阶段: {stage_name} ({current}/{total})")

    try:
        from src.core.settings import load_settings
        from src.ingestion.pipeline import IngestionPipeline

        settings = load_settings()
        pipeline = IngestionPipeline(settings)
        result = pipeline.run(
            file_path=tmp_path,
            collection=collection,
            force=force,
            on_progress=on_progress,
        )
        progress_bar.progress(1.0, text="✅ 完成！")
        status_text.success(
            f"摄取成功！文件: {uploaded_file.name}, "
            f"Chunk 数: {result.get('chunk_count', '?')}"
        )
    except Exception as e:
        progress_bar.progress(1.0, text="❌ 失败")
        status_text.error(f"摄取失败：{e}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _delete_document(data_service: DataService, source_path: str, collection: str):
    """Delete a document via DocumentManager."""
    try:
        manager = data_service._get_manager()
        result = manager.delete_document(source_path, collection)
        if result.success:
            st.success(f"已删除：{source_path}（{result.chunks_deleted} chunks, {result.images_deleted} 图片）")
            st.rerun()
        else:
            st.error(f"删除失败：{result.error}")
    except Exception as e:
        st.error(f"删除操作异常：{e}")
