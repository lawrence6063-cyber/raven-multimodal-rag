"""Qwen (通义千问) Embedding implementation — 阿里云 DashScope API.

Qwen Embedding 中文场景分析：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【模型系列】
- text-embedding-v3（最新，推荐）：1024维，支持 512/1024/1536/2048 多维度切换
- text-embedding-v2：1536维，稳定版本
- text-embedding-v1：1536维，旧版

【中文场景优势】
1. 中文语义理解极强：原生中文训练，对中文分词、成语、专业术语理解优于 OpenAI
2. 国内访问稳定：阿里云 DashScope 服务，延迟低（<100ms），无需翻墙
3. 数据合规：数据不出境，满足国内企业合规要求
4. 性价比高：text-embedding-v3 价格约 0.0007元/千token，远低于 OpenAI
5. 长文本支持：最大 8192 tokens，覆盖绝大多数 chunk 场景
6. MTEB-Chinese 排行榜表现优异：在 C-MTEB 中文基准上排名前列

【与 OpenAI 对比（中文场景）】
- 中文检索准确率：Qwen > OpenAI（尤其是专业领域、口语化表达）
- 中英混合文档：OpenAI ≈ Qwen（各有优势）
- 纯英文文档：OpenAI > Qwen
- API 兼容性：Qwen 兼容 OpenAI SDK 格式，切换成本极低

【适用场景】
- 中文为主的知识库（技术文档、产品手册、客服FAQ）
- 国内部署、数据不出境要求
- 需要低延迟的实时检索场景
- 预算敏感的中小团队

【API 说明】
Qwen Embedding 兼容 OpenAI SDK 格式，base_url 为：
https://dashscope.aliyuncs.com/compatible-mode/v1
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.libs.embedding.base_embedding import BaseEmbedding, EmbeddingError
from src.libs.embedding.embedding_factory import register_embedding

if TYPE_CHECKING:
    from src.core.settings import EmbeddingSettings


@register_embedding("qwen")
class QwenEmbedding(BaseEmbedding):
    """阿里云 Qwen (通义千问) Embedding 实现.

    通过 DashScope API 调用，兼容 OpenAI SDK 格式。
    推荐模型：text-embedding-v3（中文场景最优）。
    """

    DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # DashScope 同步 Embedding 接口单次最多 10 条文本（text-embedding-v3）。
    # 上层可按任意 batch_size 调用，本实现内部自动再切分以遵守该限制。
    MAX_BATCH_SIZE = 10

    def __init__(self, settings: "EmbeddingSettings"):
        self._settings = settings
        self._model = settings.model or "text-embedding-v3"
        self._api_key = settings.api_key
        self._base_url = settings.base_url or self.DEFAULT_BASE_URL
        self._dimensions = settings.dimensions or 1024

    def embed(self, texts: list[str]) -> list[list[float]]:
        """使用 Qwen Embedding API 将文本转换为向量.

        内部按 ``MAX_BATCH_SIZE`` 自动分批调用，以遵守 DashScope 单次 ≤10 条
        的限制；分批顺序与输入顺序严格对应。

        Args:
            texts: 待编码的文本列表。

        Returns:
            向量列表，每个向量为浮点数列表，顺序与 ``texts`` 一致。

        Raises:
            EmbeddingError: API 调用失败时抛出。
        """
        if not texts:
            return []

        import openai

        try:
            client = openai.OpenAI(api_key=self._api_key, base_url=self._base_url)
            vectors: list[list[float]] = []
            for start in range(0, len(texts), self.MAX_BATCH_SIZE):
                batch = texts[start:start + self.MAX_BATCH_SIZE]
                response = client.embeddings.create(
                    model=self._model,
                    input=batch,
                    dimensions=self._dimensions,
                )
                vectors.extend(item.embedding for item in response.data)
            return vectors
        except openai.AuthenticationError as e:
            raise EmbeddingError(
                "Qwen API 认证失败，请检查 DashScope API Key。",
                provider="qwen",
                cause=e,
            ) from e
        except Exception as e:
            raise EmbeddingError(
                f"Qwen Embedding API 调用失败: {e}",
                provider="qwen",
                cause=e,
            ) from e

    @property
    def provider_name(self) -> str:
        return "qwen"

    @property
    def dimensions(self) -> int:
        return self._dimensions
