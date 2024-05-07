import json

from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

from adapters.abstract_adapters.api_key_adapter_mixin import ApiKeyAdapterMixin
from adapters.abstract_adapters.sdk_chat_adapter import SDKChatAdapter
from adapters.types import (
    ConversationRole,
    Cost,
    OpenAIChatAdapterResponse,
    RequestBody,
    Turn,
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
    ) -> OpenAIChatAdapterResponse:
        choices = response.choices
        prompt_tokens = response.usage.prompt_tokens if response.usage else 0
        completion_tokens = response.usage.completion_tokens if response.usage else 0
        cost = (
            self.get_model().cost.prompt * prompt_tokens
            + self.get_model().cost.completion * completion_tokens
            + self.get_model().cost.request
        )

        return OpenAIChatAdapterResponse(
            response=Turn(
                role=ConversationRole.assistant,
                content=choices[0].message.content or "",
            ),  # TODO: Refactor response
            choices=choices,
            cost=cost,
            token_counts=Cost(
                prompt=prompt_tokens,
                completion=completion_tokens,
            ),
        )

    def extract_stream_response(self, request, response: ChatCompletionChunk) -> str:
        return f"data: {json.dumps(response.dict())}\n\n"
