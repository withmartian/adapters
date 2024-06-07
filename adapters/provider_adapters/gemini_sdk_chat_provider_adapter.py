import re
from typing import Any, Dict, Pattern

import google.generativeai as genai
from google.ai.generativelanguage import (
    GenerativeServiceAsyncClient,
    GenerativeServiceClient,
)
from google.api_core.client_options import ClientOptions

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

PROVIDER_NAME = "gemini"
API_KEY_NAME = "GEMINI_API_KEY"
API_KEY_PATTERN = re.compile(r".*")


class GeminiModel(Model):
    vendor_name: str = PROVIDER_NAME
    provider_name: str = PROVIDER_NAME


MODELS = [
    GeminiModel(
        name="gemini-1.0-pro",
        cost=Cost(prompt=0.125e-6, completion=0.375e-6),
        context_length=30720,
        completion_length=2048,
    ),
    GeminiModel(
        name="gemini-1.5-pro-latest",
        cost=Cost(prompt=3.5e-6, completion=10.5e-6),
        context_length=128000,
    ),
]


# TODO: max_tokens doesnt work
class GeminiSDKChatProviderAdapter(
    ProviderAdapterMixin,
    ApiKeyAdapterMixin,
    SDKChatAdapter,
):
    @staticmethod
    def get_supported_models():
        return MODELS

    @staticmethod
    def get_provider_name() -> str:
        return PROVIDER_NAME

    @staticmethod
    def get_base_sdk_url() -> str:
        return ""

    @staticmethod
    def get_api_key_name() -> str:
        return API_KEY_NAME

    @staticmethod
    def get_api_key_pattern() -> Pattern:
        return API_KEY_PATTERN

    _sync_client: GenerativeServiceClient
    _async_client: GenerativeServiceAsyncClient

    def __init__(
        self,
    ):
        super().__init__()
        self.set_api_key(self.get_api_key())

    def get_model_name(self) -> str:
        if self._current_model is None:
            raise ValueError("Model not set")
        return self._current_model.name

    def get_async_client(self):
        return self._async_client

    def get_sync_client(self):
        return self._sync_client

    def adjust_temperature(self, temperature: float) -> float:
        return temperature / 2

    def set_api_key(self, api_key: str) -> None:
        super().set_api_key(api_key)

        self._sync_client = GenerativeServiceClient(
            client_options=ClientOptions(api_key=api_key), transport="rest"
        )
        self._async_client = GenerativeServiceAsyncClient(
            client_options=ClientOptions(api_key=api_key), transport="rest"
        )

    def extract_response(
        self, request: Any, response: Any
    ) -> OpenAIChatAdapterResponse:
        model = genai.GenerativeModel(model_name=self.get_model_name())
        model._client = self.get_sync_client()

        choices = [
            {
                "message": {
                    "role": ConversationRole.assistant,
                    "content": response.text,
                },
                "finish_reason": "stop",
            }
        ]

        # Optimize token count calculation, use async for async and parallelize
        prompt_tokens = model.count_tokens(
            [turn.content for turn in request.turns]
        ).total_tokens
        completion_tokens = model.count_tokens(response.text).total_tokens

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

    async def extract_response_async(
        self, request: Any, response: Any
    ) -> OpenAIChatAdapterResponse:
        model = genai.GenerativeModel(model_name=self.get_model_name())
        model._client = self.get_async_client()

        choices = [
            {
                "message": {
                    "role": ConversationRole.assistant,
                    "content": response.text,
                },
                "finish_reason": "stop",
            }
        ]

        # Optimize token count calculation, use async for async and parallelize
        prompt_tokens = await model.count_tokens_async(
            [turn.content for turn in request.turns]
        ).total_tokens
        completion_tokens = await model.count_tokens_async(response.text).total_tokens

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

    def extract_stream_response(self, request, response):
        return response

    def get_params(
        self,
        llm_input: Conversation,
        **kwargs,
    ) -> Dict[str, Any]:
        params = super().get_params(llm_input, **kwargs)

        last_message = params["messages"].pop(-1)["content"]

        messages = [
            {
                **({"parts": [message["content"]]}),
                **(
                    {"role": ConversationRole.user}
                    if message["role"] == ConversationRole.system
                    else {"role": message["role"]}
                ),
            }
            for message in params["messages"]
        ]

        if "max_tokens" in params:
            params["max_output_tokens"] = params["max_tokens"]
            del params["max_tokens"]

        del params["messages"]

        return {"config": params, "history": messages, "prompt": last_message}

    async def execute_async(
        self,
        llm_input: Conversation,
        **kwargs,
    ):
        params = self.get_params(llm_input, **kwargs)

        model = genai.GenerativeModel(
            model_name=self.get_model_name(),
            generation_config=params["config"],
        )
        model._async_client = self.get_async_client()

        convo = model.start_chat(history=params["history"])

        result = await convo.send_message_async(params["prompt"])

        return await self.extract_response_async(request=llm_input, response=result)

    def execute_sync(
        self,
        llm_input: Conversation,
        **kwargs,
    ):
        params = self.get_params(llm_input, **kwargs)

        model = genai.GenerativeModel(
            model_name=self.get_model_name(), generation_config=params["config"]
        )
        model._client = self.get_sync_client()

        convo = model.start_chat(history=params["history"])

        result = convo.send_message(params["prompt"])

        return self.extract_response(request=llm_input, response=result)
