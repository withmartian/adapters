import json
import re
from typing import Any, Pattern

from cohere import AsyncClient, Client

from adapters.abstract_adapters.api_key_adapter_mixin import ApiKeyAdapterMixin
from adapters.abstract_adapters.provider_adapter_mixin import ProviderAdapterMixin
from adapters.abstract_adapters.sdk_chat_adapter import SDKChatAdapter
from adapters.types import (
    Conversation,
    ConversationRole,
    Cost,
    Model,
    OpenAIChatAdapterResponse,
    Turn,
)

API_KEY_NAME = "COHERE_API_KEY"
API_KEY_PATTERN = re.compile(r".*")
BASE_URL = "https://api.cohere.ai/v1"
PROVIDER_NAME = "cohere"


class CohereModel(Model):
    supports_streaming: bool = True
    supports_json_content: bool = True
    vendor_name: str = PROVIDER_NAME
    provider_name: str = PROVIDER_NAME

    def _get_api_path(self) -> str:
        return self.name


MODELS = [
    CohereModel(
        name="command-r",
        cost=Cost(prompt=0.5e-6, completion=1.5e-6),
        context_length=131_072,
    ),
    CohereModel(
        name="command-r-plus",
        cost=Cost(prompt=3.00e-6, completion=15.00e-6),
        context_length=131_072,
    ),
]

FINISH_REASON_MAPPING = {
    "COMPLETE": "stop",
    "MAX_TOKENS": "length",
}

ROLE_MAPPING = {
    "user": "USER",
    "assistant": "CHATBOT",
    "system": "SYSTEM",
    "USER": "user",
    "CHATBOT": "assistant",
    "SYSTEM": "system",
}


class CohereSDKChatProviderAdapter(
    ProviderAdapterMixin,
    ApiKeyAdapterMixin,
    SDKChatAdapter,
):
    @staticmethod
    def get_api_key_name() -> str:
        return API_KEY_NAME

    @staticmethod
    def get_api_key_pattern() -> Pattern:
        return API_KEY_PATTERN

    def get_base_sdk_url(self) -> str:
        return BASE_URL

    @staticmethod
    def get_provider_name() -> str:
        return PROVIDER_NAME

    @staticmethod
    def get_supported_models():
        return MODELS

    _sync_client: Client
    _async_client: AsyncClient

    def __init__(
        self,
    ):
        super().__init__()
        self._sync_client = Client(
            api_key=self.get_api_key(), base_url=self.get_base_sdk_url()
        )
        self._async_client = AsyncClient(
            api_key=self.get_api_key(), base_url=self.get_base_sdk_url()
        )

    async def _async_client_wrapper(self, **kwargs: Any):
        stream = kwargs.get("stream", False)

        if "stream" in kwargs:
            del kwargs["stream"]

        if stream:
            # Cohere uses a "sync" call to chat_stream, even if it is an async_client.
            return self._async_client.chat_stream(**kwargs)

        return await self._async_client.chat(**kwargs)

    def _sync_client_wrapper(self, **kwargs: Any):
        stream = kwargs.get("stream", False)

        if "stream" in kwargs:
            del kwargs["stream"]

        if stream:
            return self._sync_client.chat_stream(**kwargs)

        return self._sync_client.chat(**kwargs)

    def set_api_key(self, api_key: str) -> None:
        super().set_api_key(api_key)

        # Using internal variables to set the api_key
        self._sync_client._client_wrapper._token = api_key
        self._async_client._client_wrapper._token = api_key

    def get_async_client(self):
        return self._async_client_wrapper

    def get_sync_client(self):
        return self._sync_client_wrapper

    def get_params(self, llm_input: Conversation, **kwargs: Any) -> dict[str, Any]:
        params = super().get_params(llm_input, **kwargs)

        for message in params["messages"]:
            # Use content as message
            message["message"] = message["content"]
            del message["content"]

            # Map role to Cohere's role
            message["role"] = ROLE_MAPPING.get(message["role"])

            # Join content if it's a list
            if isinstance(message["message"], list):
                message["message"] = " ".join(
                    content.get("text", "") for content in message["message"]
                )

            # Cohere doesn't allow empty strings
            if message["message"] == "":
                message["message"] = " "

        last_message = params["messages"][-1]

        params["chat_history"] = params["messages"][:-1]
        del params["messages"]

        params["message"] = last_message["message"]

        return params

    def extract_response(
        self, request: Any, response: Any
    ) -> OpenAIChatAdapterResponse:
        choices = [
            {
                "message": {
                    "role": "assistant",
                    "content": response.text,
                },
                "finish_reason": FINISH_REASON_MAPPING.get(response.finish_reason),
            }
        ]

        prompt_tokens = response.meta.tokens.input_tokens
        completion_tokens = response.meta.tokens.output_tokens
        cost = (
            self.get_model().cost.prompt * prompt_tokens
            + self.get_model().cost.completion * completion_tokens
            + self.get_model().cost.request
        )

        return OpenAIChatAdapterResponse(
            response=Turn(
                role=ConversationRole.assistant,
                content=choices[0]["message"]["content"],  # type: ignore
            ),  # TODO: Refactor response
            choices=choices,
            cost=cost,
            token_counts=Cost(
                prompt=prompt_tokens,
                completion=completion_tokens,
            ),
        )

    def extract_stream_response(self, request: Any, response: Any) -> str:
        content = getattr(response, "text", "")
        if response.event_type == "stream-end":
            content = None

        chunk = json.dumps(
            {
                "choices": [
                    {
                        "delta": {
                            "role": ConversationRole.assistant,
                            "content": content,
                        },
                    }
                ]
            }
        )

        return f"data: {chunk}\n\n"
