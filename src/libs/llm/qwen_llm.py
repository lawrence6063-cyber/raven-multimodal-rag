"""Qwen (通义千问) LLM implementation — OpenAI-compatible DashScope API.

Qwen 对话模型通过阿里云 DashScope 提供，兼容 OpenAI SDK 格式，仅 base_url 不同：
https://dashscope.aliyuncs.com/compatible-mode/v1

与 ``QwenEmbedding`` 复用同一个 DashScope API Key，即可用单个 key 同时驱动
生成（LLM）与向量化（Embedding）两条链路。常用模型：qwen-plus / qwen-max /
qwen-turbo。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.libs.llm.llm_factory import register_llm
from src.libs.llm.openai_llm import OpenAILLM
from src.libs.throttle import dashscope_limiter, is_dashscope_throttling, retry_call

if TYPE_CHECKING:
    from src.core.settings import LLMSettings
    from src.libs.llm.base_llm import ChatMessage, ChatResponse


@register_llm("qwen")
class QwenLLM(OpenAILLM):
    """Qwen LLM via Alibaba DashScope (OpenAI-compatible, just a different base_url)."""

    DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def __init__(self, settings: "LLMSettings"):
        super().__init__(settings)
        self._base_url = settings.base_url or self.DEFAULT_BASE_URL

    def chat(self, messages: "list[ChatMessage]", **kwargs) -> "ChatResponse":
        """Chat with proactive rate limiting + backoff retry on DashScope throttling."""
        dashscope_limiter.wait()
        return retry_call(
            lambda: OpenAILLM.chat(self, messages, **kwargs),
            is_retryable=is_dashscope_throttling,
        )

    @property
    def provider_name(self) -> str:
        return "qwen"
