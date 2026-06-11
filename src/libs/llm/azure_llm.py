"""Azure OpenAI LLM implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.libs.llm.base_llm import BaseLLM, ChatMessage, ChatResponse, LLMError
from src.libs.llm.llm_factory import register_llm

if TYPE_CHECKING:
    from src.core.settings import LLMSettings


@register_llm("azure")
class AzureLLM(BaseLLM):
    """Azure OpenAI Service LLM implementation."""

    def __init__(self, settings: "LLMSettings"):
        self._settings = settings
        self._model = settings.deployment_name or settings.model
        self._api_key = settings.api_key
        self._azure_endpoint = settings.azure_endpoint
        self._api_version = settings.api_version
        self._temperature = settings.temperature
        self._max_tokens = settings.max_tokens

    def chat(self, messages: list[ChatMessage], **kwargs) -> ChatResponse:
        """Send chat request to Azure OpenAI."""
        import openai

        try:
            client = openai.AzureOpenAI(
                api_key=self._api_key,
                azure_endpoint=self._azure_endpoint,
                api_version=self._api_version,
            )
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
            raise LLMError("Authentication failed. Check Azure API key.", provider="azure", cause=e) from e
        except openai.APIConnectionError as e:
            raise LLMError(f"Connection error: {e}", provider="azure", cause=e) from e
        except Exception as e:
            raise LLMError(f"API call failed: {e}", provider="azure", cause=e) from e

    @property
    def provider_name(self) -> str:
        return "azure"
