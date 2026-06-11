"""Semantic Splitter — 基于 Embedding 向量的语义分割。

================================================================================
📖 Semantic Split 原理说明
================================================================================

【核心思想】
Semantic Split 不依赖固定分隔符，而是通过 Embedding 模型计算相邻文本片段的
语义相似度，在"语义断裂点"（相似度骤降处）进行切分。

【是否需要大模型（LLM）？】
❌ 不需要 LLM（大语言模型/生成式模型）
✅ 只需要 Embedding 模型（向量编码模型）

Embedding 模型的推理成本远低于 LLM：
- Qwen text-embedding-v3: ¥0.0005/千token（约为 LLM 的 1/4000）
- 一篇 5000 token 文档做语义分割，成本不到 3 厘钱

【Qwen Embedding 接口支持】
阿里云 DashScope 提供完整的 Embedding API，兼容 OpenAI SDK 格式：
- text-embedding-v4 (最新): 2048维可选，8192 max tokens，¥0.0005/千token
- text-embedding-v3 (推荐): 1024维可选，8192 max tokens，¥0.0005/千token
- text-embedding-v2 (稳定): 1536维，2048 max tokens，¥0.0007/千token
- 免费额度：各模型 100万 Token（开通后90天有效）

【算法步骤】
1. 按句子/段落对文本进行初步切分（得到 sentences）
2. 对每个 sentence 调用 Embedding 模型生成向量
3. 计算相邻 sentence 向量的余弦相似度
4. 设定阈值（或使用百分位/标准差策略），找到相似度骤降的位置
5. 在断裂点处切分，将连续的高相似度句子合并为一个 chunk
6. 对超长 chunk 进行二次切分（确保不超过 chunk_size）

【适用场景】
- 语义边界不规则的长文本（如会议记录、访谈、论文）
- 对切分质量要求极高的场景
- 文档结构不明显（无标题、无段落分隔）的纯文本

【局限性】
- 需要调用 Embedding API，有网络延迟和费用
- 处理速度慢于 Recursive Splitter（约 10-50x）
- 对短文档效果不明显
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np

from src.libs.splitter.base_splitter import BaseSplitter, SplitterError
from src.libs.splitter.splitter_factory import register_splitter

if TYPE_CHECKING:
    from src.core.settings import SplitterSettings
    from src.libs.embedding.base_embedding import BaseEmbedding


@register_splitter("semantic")
class SemanticSplitter(BaseSplitter):
    """基于 Embedding 语义相似度的文本分割器。

    通过计算相邻句子的向量余弦相似度，在语义断裂点进行切分。
    支持任意 Embedding Provider（推荐 Qwen text-embedding-v3 用于中文场景）。
    """

    # 用于初步句子切分的正则（中英文句号、问号、感叹号、换行）
    SENTENCE_PATTERN = re.compile(r'(?<=[。！？.!?\n])\s*')

    def __init__(self, settings: "SplitterSettings", embedding: "BaseEmbedding | None" = None):
        """初始化 Semantic Splitter。

        Args:
            settings: Splitter 配置（chunk_size, chunk_overlap）。
            embedding: Embedding 实例，用于生成向量。
                       如果为 None，将在首次调用时通过工厂创建。
        """
        self._chunk_size = settings.chunk_size
        self._chunk_overlap = settings.chunk_overlap
        self._embedding = embedding
        # 语义断裂阈值：使用百分位策略（默认取相似度最低的 25% 作为断裂点）
        self._breakpoint_percentile = 25

    def _get_embedding(self) -> "BaseEmbedding":
        """懒加载 Embedding 实例。"""
        if self._embedding is None:
            try:
                from src.core.settings import Settings
                from src.libs.embedding.embedding_factory import EmbeddingFactory

                settings = Settings.load()
                self._embedding = EmbeddingFactory.create(settings.embedding)
            except Exception as e:
                raise SplitterError(
                    f"无法创建 Embedding 实例: {e}。"
                    "请通过构造函数传入 embedding 参数，或确保 embedding 配置正确。",
                    provider="semantic",
                ) from e
        return self._embedding

    def _split_sentences(self, text: str) -> list[str]:
        """将文本按句子边界初步切分。"""
        sentences = self.SENTENCE_PATTERN.split(text)
        # 过滤空句子，合并过短的句子
        result = []
        buffer = ""
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            buffer += s
            # 至少积累 50 个字符再作为一个独立句子（避免过度碎片化）
            if len(buffer) >= 50:
                result.append(buffer)
                buffer = ""
        if buffer:
            if result:
                result[-1] += buffer
            else:
                result.append(buffer)
        return result

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """计算两个向量的余弦相似度。"""
        a_arr = np.array(a)
        b_arr = np.array(b)
        dot = np.dot(a_arr, b_arr)
        norm = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
        if norm == 0:
            return 0.0
        return float(dot / norm)

    def _find_breakpoints(self, similarities: list[float]) -> list[int]:
        """找到语义断裂点（相似度骤降的位置）。

        使用百分位策略：相似度低于第 N 百分位的位置视为断裂点。
        """
        if not similarities:
            return []

        threshold = float(np.percentile(similarities, self._breakpoint_percentile))
        breakpoints = []
        for i, sim in enumerate(similarities):
            if sim < threshold:
                breakpoints.append(i)
        return breakpoints

    def _merge_chunks(self, sentences: list[str], breakpoints: list[int]) -> list[str]:
        """根据断裂点将句子合并为 chunk，并处理超长 chunk。"""
        chunks = []
        current_chunk_sentences: list[str] = []

        for i, sentence in enumerate(sentences):
            current_chunk_sentences.append(sentence)

            # 如果当前位置是断裂点，或者是最后一个句子
            if i in breakpoints or i == len(sentences) - 1:
                chunk_text = " ".join(current_chunk_sentences)

                # 如果 chunk 超过 chunk_size，进行二次切分
                if len(chunk_text) > self._chunk_size:
                    sub_chunks = self._force_split(chunk_text)
                    chunks.extend(sub_chunks)
                else:
                    chunks.append(chunk_text)

                current_chunk_sentences = []

        return chunks

    def _force_split(self, text: str) -> list[str]:
        """对超长文本进行强制切分（按 chunk_size 滑动窗口）。"""
        chunks = []
        start = 0
        while start < len(text):
            end = start + self._chunk_size
            chunk = text[start:end]
            chunks.append(chunk.strip())
            start = end - self._chunk_overlap
        return [c for c in chunks if c]

    def split_text(self, text: str) -> list[str]:
        """使用语义相似度对文本进行分割。

        Args:
            text: 待分割的文本。

        Returns:
            分割后的文本块列表。

        Raises:
            SplitterError: 分割失败时抛出。
        """
        if not text or not text.strip():
            return []

        try:
            # Step 1: 按句子初步切分
            sentences = self._split_sentences(text)

            if len(sentences) <= 1:
                return sentences

            # Step 2: 获取所有句子的 Embedding 向量
            embedding = self._get_embedding()
            vectors = embedding.embed(sentences)

            # Step 3: 计算相邻句子的余弦相似度
            similarities = []
            for i in range(len(vectors) - 1):
                sim = self._cosine_similarity(vectors[i], vectors[i + 1])
                similarities.append(sim)

            # Step 4: 找到语义断裂点
            breakpoints = self._find_breakpoints(similarities)

            # Step 5: 根据断裂点合并为 chunk
            chunks = self._merge_chunks(sentences, breakpoints)

            return [c for c in chunks if c.strip()]

        except SplitterError:
            raise
        except Exception as e:
            raise SplitterError(
                f"语义分割失败: {e}",
                provider="semantic",
            ) from e

    @property
    def provider_name(self) -> str:
        return "semantic"
