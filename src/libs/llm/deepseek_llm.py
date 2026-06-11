"""DeepSeek LLM implementation — OpenAI-compatible API."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.libs.llm.openai_llm import OpenAILLM
from src.libs.llm.llm_factory import register_llm

if TYPE_CHECKING:
    from src.core.settings import LLMSettings


@register_llm("deepseek")
class DeepSeekLLM(OpenAILLM):
    """DeepSeek LLM — OpenAI-compatible, just different base_url."""

    DEFAULT_BASE_URL = "https://api.deepseek.com/v1"

    def __init__(self, settings: "LLMSettings"):
        super().__init__(settings)
        self._base_url = settings.base_url or self.DEFAULT_BASE_URL

    @property
    def provider_name(self) -> str:
        return "deepseek"
