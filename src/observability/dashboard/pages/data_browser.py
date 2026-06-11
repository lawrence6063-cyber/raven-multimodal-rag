"""Data Browser page — browse documents, chunks, and images (G3)."""

from __future__ import annotations

import streamlit as st

from src.observability.dashboard.services.data_service import DataService


def render():
    """Render the data browser page."""
    st.title("📂 数据浏览器")
    st.markdown("浏览已摄入的文档、Chunk 详情和关联图片。")

    data_service = DataService()

    # Collection filter
    try:
        collections = data_service.list_collections()
    except Exception:
        collections = []

    col1, col2 = st.columns([1, 3])
    with col1:
        options = ["全部"] + collections
        selected = st.selectbox("集合筛选", options)

    collection_filter = None if selected == "全部" else selected

    # Document list
    st.subheader("📄 文档列表")
    try:
        docs = data_service.list_documents(collection_filter)
    except Exception as e:
        st.error(f"加载文档列表失败：{e}")
        return

    if not docs:
        st.info("暂无已摄入的文档。请先通过 Ingestion 管理页面上传文件。")
        return

    for doc in docs:
        with st.expander(
            f"📄 {doc.source_path}  |  集合: {doc.collection}  |  "
            f"Chunks: {doc.chunk_count}  |  图片: {doc.image_count}  |  "
            f"时间: {doc.processed_at}"
        ):
            st.markdown(f"**文件哈希**: `{doc.file_hash}`")

            # Try to load chunk details
            if st.button(f"查看 Chunks", key=f"chunks_{doc.file_hash}"):
                _show_chunks(data_service, doc)


def _show_chunks(data_service: DataService, doc):
    """Show chunk details for a document."""
    # We need doc_id from chroma metadata; use file_hash as proxy query
    st.markdown("---")
    st.markdown("#### Chunk 列表")
    st.info("Chunk 详情需通过 doc_id 查询。请在 MCP Server 的 get_document_summary tool 中使用。")
