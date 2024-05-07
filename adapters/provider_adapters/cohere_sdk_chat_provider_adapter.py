import json
import re
from collections.abc import AsyncIterator, Callable
from functools import partial
from typing import Any, Pattern

from cohere import (  # type: ignore[import-not-found]
    AsyncClient,
    Client,
    NonStreamedChatResponse,
)

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
BASE_URL = "https://api.cohere.ai/v1/chat"
PROVIDER_NAME = "cohere"


class CohereModel(Model):
    supports_streaming: bool = True
    supports_json_content: bool = True
    vendor_name: str = PROVIDER_NAME
    provider_name: str = PROVIDER_NAME


MODELS = [
    CohereModel(
        name="command-r",
        cost=Cost(prompt=3.00e-6, completion=15.00e-6),
        context_length=131_072,
    ),
    CohereModel(
        name="command-r-plus",
        cost=Cost(prompt=0.50e-6, completion=1.50e-6),
        context_length=131_072,
    ),
]

FINISH_REASON_MAPPING = {
    "COMPLETE": "stop",
    "MAX_TOKENS": "length",
}

MAP_CONVERSATION_ROLE_TO_COHERE = {
    "user": "USER",
    "assistant": "CHATBOT",
    "system": "SYSTEM",
}
MAP_CONVERSATION_ROLE_TO_OPENAI = {
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

    @staticmethod
    def get_base_sdk_url() -> str:
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
        self._sync_client = Client(api_key=self.get_api_key())
        self._async_client = AsyncClient(api_key=self.get_api_key())
        self.params = None
        self.stream = None
        self.async_chat_stream = False

    def get_params(self, llm_input: Conversation, **kwargs: Any) -> dict[str, Any]:
        params = super().get_params(llm_input, **kwargs)
        for message in params["messages"]:
            message["role"] = MAP_CONVERSATION_ROLE_TO_COHERE.get(message["role"])
            if isinstance(message["content"], list):
                message["message"] = " ".join(
                    content.get("text", "") for content in message["content"]
                )
            elif not message["content"]:  # Empty string
                message["message"] = " "
            else:
                message["message"] = message["content"]

            del message["content"]

        last_message = params["messages"][-1]
        params["chat_history"] = params["messages"][:-1]
        params["message"] = last_message["message"]
        del params["messages"]
        self.params = params

        self.stream = params.get("stream", False)
        if self.stream:  # Stream not allowed in the Cohere's sdk.
            del params["stream"]
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

        # return_prompt=True and response.prompt not working.
        for choice in self.params["chat_history"]:
            choices.append(
                {
                    "message": {
                        "role": MAP_CONVERSATION_ROLE_TO_OPENAI[choice["role"]],
                        "content": choice["message"],
                    }
                }
            )

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

    def get_async_client(self) -> Callable[..., AsyncIterator[NonStreamedChatResponse]]:
        if not self.stream:
            return partial(self._async_client.chat, return_prompt=True)
        self.async_chat_stream = True
        return partial(self._async_client.chat_stream, return_prompt=True)

    def get_sync_client(self) -> Callable[..., NonStreamedChatResponse]:
        if not self.stream:
            return partial(self._sync_client.chat, return_prompt=True)
        return partial(self._sync_client.chat_stream, return_prompt=True)

    def get_model_name(self) -> str:
        if self._current_model is None:
            raise ValueError("Model not set")
        return self._current_model.name

    def set_api_key(self, api_key: str) -> None:
        super().set_api_key(api_key)

        self._sync_client.api_key = api_key
        self._async_client.api_key = api_key
