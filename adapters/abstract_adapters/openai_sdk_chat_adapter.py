from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

from adapters.abstract_adapters.api_key_adapter_mixin import ApiKeyAdapterMixin
from adapters.abstract_adapters.sdk_chat_adapter import SDKChatAdapter
from adapters.types import (
    AdapterChatCompletion,
    AdapterChatCompletionChunk,
    RequestBody,
)
from adapters.adapter_factory import _client_cache


class OpenAISDKChatAdapter(SDKChatAdapter):
    _sync_client: OpenAI
    _async_client: AsyncOpenAI

    def __init__(
        self,
    ):
        super().__init__()
        self._sync_client = OpenAI(
            api_key=self.get_api_key(),
            base_url=self.get_base_sdk_url(),
        )
        self._async_client = AsyncOpenAI(
            api_key=self.get_api_key(),
            base_url=self.get_base_sdk_url(),
        )

    def _call_sync(self):
        return self._sync_client.chat.completions.create

    def _call_async(self):
        return self._async_client.chat.completions.create

    def _client_sync(self, base_url: str, api_key: str):
        return OpenAI(
            base_url=base_url,
            api_key=api_key,
        )

    def _client_async(self, base_url: str, api_key: str):
        return AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
        )

    def set_api_key(self, api_key: str) -> None:
        super().set_api_key(api_key)

        cached_client_sync_path = f"{self.get_base_sdk_url()}-{api_key}-sync"
        cached_client_async_path = f"{self.get_base_sdk_url()}-{api_key}-async"

        if not _client_cache.get(cached_client_sync_path):
            _client_cache[cached_client_sync_path] = OpenAI(
                api_key=api_key,
                base_url=self.get_base_sdk_url(),
            )

        if not _client_cache.get(cached_client_async_path):
            _client_cache[cached_client_async_path] = AsyncOpenAI(
                api_key=api_key,
                base_url=self.get_base_sdk_url(),
            )

        self._sync_client = _client_cache[cached_client_sync_path]
        self._async_client = _client_cache[cached_client_async_path]

    def _extract_response(
        self,
        request: RequestBody,
        response: ChatCompletion,
    ) -> AdapterChatCompletion:
        prompt_tokens = float(response.usage.prompt_tokens if response.usage else 0)
        completion_tokens = float(
            response.usage.completion_tokens if response.usage else 0
        )
        reasoning_tokens = float(
            response.usage.completion_tokens_details.reasoning_tokens
            if response.usage
            and response.usage.completion_tokens_details
            and response.usage.completion_tokens_details.reasoning_tokens
            else 0
        )

        cost = (
            self.get_model().cost.prompt * prompt_tokens
            + self.get_model().cost.completion * completion_tokens
            + reasoning_tokens * completion_tokens
            + self.get_model().cost.request
        )

        return AdapterChatCompletion.model_construct(
            **response.model_dump(),
            cost=cost,
        )

    def _extract_stream_response(
        self, request, response: ChatCompletionChunk, state: dict
    ) -> AdapterChatCompletionChunk:
        return AdapterChatCompletionChunk.model_construct(
            **response.model_dump(),
        )
