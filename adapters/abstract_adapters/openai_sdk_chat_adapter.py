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
from adapters.utils.openai_client_factory import OpenAIClientFactory


class OpenAISDKChatAdapter(ApiKeyAdapterMixin, SDKChatAdapter):
    _sync_client: OpenAI
    _async_client: AsyncOpenAI

    def __init__(
        self,
    ):
        super().__init__()
        self._sync_client = OpenAIClientFactory.get_openai_sync_client(
            api_key=self.get_api_key(),
            base_url=self.get_base_sdk_url(),
        )
        self._async_client = OpenAIClientFactory.get_openai_async_client(
            api_key=self.get_api_key(),
            base_url=self.get_base_sdk_url(),
        )

    def get_async_client(self):
        return self._async_client.chat.completions.create

    def get_sync_client(self):
        return self._sync_client.chat.completions.create

    def set_api_key(self, api_key: str) -> None:
        super().set_api_key(api_key)

        self._sync_client.api_key = api_key
        self._async_client.api_key = api_key

    def extract_response(
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

    def extract_stream_response(
        self, request, response: ChatCompletionChunk, state: dict
    ) -> AdapterChatCompletionChunk:
        return AdapterChatCompletionChunk.model_construct(
            **response.model_dump(),
        )
