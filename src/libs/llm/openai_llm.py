"""OpenAI LLM implementation — supports OpenAI API directly."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.libs.llm.base_llm import BaseLLM, ChatMessage, ChatResponse, LLMError
from src.libs.llm.llm_factory import register_llm

if TYPE_CHECKING:
    from src.core.settings import LLMSettings


@register_llm("openai")
class OpenAILLM(BaseLLM):
    """OpenAI API LLM implementation."""

    DEFAULT_BASE_URL = "https://api.openai.com/v1"

    def __init__(self, settings: "LLMSettings"):
        self._settings = settings
        self._model = settings.model
        self._api_key = settings.api_key
        self._base_url = settings.base_url or self.DEFAULT_BASE_URL
        self._temperature = settings.temperature
        self._max_tokens = settings.max_tokens

    def chat(self, messages: list[ChatMessage], **kwargs) -> ChatResponse:
        """Send chat request to OpenAI API."""
        import openai

        try:
            client = openai.OpenAI(api_key=self._api_key, base_url=self._base_url)
            response = client.chat.completions.create(
                model=self._model,
                messages=[{"role": m.role, "content": m.content} for m in messages],
                temperature=kwargs.get("temperature", self._temperature),
                max_tokens=kwargs.get("max_tokens", self._max_tokens),
            )
            choice = response.choices[0]
            usage = {}
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
            return ChatResponse(
                content=choice.message.content or "",
                model=response.model,
                usage=usage,
                raw=response,
            )
        except openai.AuthenticationError as e:
            raise LLMError("Authentication failed. Check your API key.", provider="openai", cause=e) from e
        except openai.RateLimitError as e:
            raise LLMError("Rate limit exceeded.", provider="openai", cause=e) from e
        except openai.APIConnectionError as e:
            raise LLMError(f"Connection error: {e}", provider="openai", cause=e) from e
        except Exception as e:
            raise LLMError(f"API call failed: {e}", provider="openai", cause=e) from e

    @property
    def provider_name(self) -> str:
        return "openai"
