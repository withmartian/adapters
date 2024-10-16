from enum import Enum
import time
from typing import Any, Dict

from cohere import AsyncClientV2, ClientV2
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from adapters.abstract_adapters.api_key_adapter_mixin import ApiKeyAdapterMixin
from adapters.abstract_adapters.provider_adapter_mixin import ProviderAdapterMixin
from adapters.abstract_adapters.sdk_chat_adapter import SDKChatAdapter
from adapters.types import (
    AdapterChatCompletion,
    AdapterChatCompletionChunk,
    AdapterFinishReason,
    Conversation,
    ConversationRole,
    Cost,
    Model,
    ModelProperties,
)

API_KEY_NAME = "COHERE_API_KEY"
BASE_URL = "https://api.cohere.com"  # Updated to v2 endpoint
PROVIDER_NAME = "cohere"
BASE_PROPERTIES = ModelProperties(open_source=True, gdpr_compliant=True)


class CohereModel(Model):
    vendor_name: str = PROVIDER_NAME
    provider_name: str = PROVIDER_NAME
    properties: ModelProperties = BASE_PROPERTIES

    supports_repeating_roles: bool = True
    supports_system: bool = True
    supports_multiple_system: bool = True
    supports_empty_content: bool = True
    supports_tool_choice_required: bool = True
    supports_last_assistant: bool = True
    supports_first_assistant: bool = True
    supports_json_content: bool = True
    supports_streaming: bool = False

    def _get_api_path(self) -> str:
        return self.name


MODELS = [
    CohereModel(
        name="command-r-plus-08-2024",
        cost=Cost(prompt=0.5e-6, completion=1.5e-6),
        context_length=128000,
        properties=BASE_PROPERTIES.model_copy(update={"is_nsfw": True}),
    ),
    CohereModel(
        name="command-r-plus",
        cost=Cost(prompt=3.00e-6, completion=15.00e-6),
        context_length=128000,
        properties=BASE_PROPERTIES.model_copy(update={"is_nsfw": True}),
    ),
]


class CohereFinishReason(str, Enum):
    complete = "COMPLETE"
    max_tokens = "MAX_TOKENS"
    stop_sequence = "STOP_SEQUENCE"
    tool_call = "TOOL_CALL"
    error = "ERROR"


FINISH_REASON_MAPPING: Dict[CohereFinishReason, AdapterFinishReason] = {
    CohereFinishReason.complete: AdapterFinishReason.stop,
    CohereFinishReason.max_tokens: AdapterFinishReason.length,
    CohereFinishReason.stop_sequence: AdapterFinishReason.stop,
    CohereFinishReason.tool_call: AdapterFinishReason.tool_calls,
}


class CohereSDKChatProviderAdapter(
    ProviderAdapterMixin,
    ApiKeyAdapterMixin,
    SDKChatAdapter,
):
    @staticmethod
    def get_api_key_name() -> str:
        return API_KEY_NAME

    def get_base_sdk_url(self) -> str:
        return BASE_URL

    @staticmethod
    def get_provider_name() -> str:
        return PROVIDER_NAME

    @staticmethod
    def get_supported_models():
        return MODELS

    _sync_client: ClientV2
    _async_client: AsyncClientV2

    def __init__(
        self,
    ):
        super().__init__()
        self._sync_client = ClientV2(
            api_key=self.get_api_key(), base_url=self.get_base_sdk_url()
        )
        self._async_client = AsyncClientV2(
            api_key=self.get_api_key(), base_url=self.get_base_sdk_url()
        )

    async def _async_client_wrapper(self, **kwargs: Any):
        stream = kwargs.pop("stream", False)
        if stream:
            return self._async_client.chat_stream(**kwargs)
        return await self._async_client.chat(**kwargs)

    def _sync_client_wrapper(self, **kwargs: Any):
        stream = kwargs.pop("stream", False)

        if stream:
            return self._sync_client.chat_stream(**kwargs)
        return self._sync_client.chat(**kwargs)

    def set_api_key(self, api_key: str) -> None:
        super().set_api_key(api_key)
        self._sync_client._client_wrapper._token = api_key
        self._async_client._client_wrapper._token = api_key

    def get_async_client(self):
        return self._async_client_wrapper

    def get_sync_client(self):
        return self._sync_client_wrapper

    def get_params(self, llm_input: Conversation, **kwargs: Any) -> dict[str, Any]:
        params = super().get_params(llm_input, **kwargs)

        messages = []
        for message in params["messages"]:
            new_message = {
                "role": message["role"],
                "content": message["content"],
            }

            if isinstance(new_message["content"], list):
                new_message["content"] = " ".join(
                    content.get("text", "") for content in new_message["content"]
                )

            if new_message["content"] == "":
                new_message["content"] = " "

            messages.append(new_message)

        params["messages"] = messages
        return params

    def extract_response(self, request: Any, response: Any) -> AdapterChatCompletion:
        prompt_tokens = (
            float(response.usage.billed_units.input_tokens)
            if response.usage and hasattr(response.usage, "billed_units")
            else 0
        )
        completion_tokens = (
            float(response.usage.billed_units.output_tokens)
            if response.usage and hasattr(response.usage, "billed_units")
            else 0
        )
        cost = (
            self.get_model().cost.prompt * prompt_tokens
            + self.get_model().cost.completion * completion_tokens
            + self.get_model().cost.request
        )

        finish_reason = FINISH_REASON_MAPPING.get(
            CohereFinishReason(response.finish_reason), AdapterFinishReason.stop
        )

        choices: list[Choice] = []
        for content in response.message.content:
            if content.type == "text":
                choices.append(
                    Choice(
                        index=len(choices),
                        finish_reason=finish_reason.value,
                        message=ChatCompletionMessage(
                            role=ConversationRole.assistant.value,
                            content=content.text,
                        ),
                    )
                )
        usage = CompletionUsage(
            prompt_tokens=response.usage.billed_units.input_tokens,
            completion_tokens=response.usage.billed_units.output_tokens,
            total_tokens=response.usage.billed_units.input_tokens
            + response.usage.billed_units.output_tokens,
        )

        return AdapterChatCompletion(
            id=response.id,
            created=int(time.time()),
            model=self.get_model().name,
            object="chat.completion",
            cost=cost,
            usage=usage,
            choices=choices,
        )

    def extract_stream_response(
        self, request: Any, response: Any, state
    ) -> AdapterChatCompletionChunk:
        raise NotImplementedError
        # content = None
        # if response.type == "content-delta":
        #     content = response.delta.message.content.text
        # elif response.type == "stream-end":
        #     content = None

        # chunk = json.dumps(
        #     {
        #         "choices": [
        #             {
        #                 "delta": {
        #                     "role": ConversationRole.assistant,
        #                     "content": content,
        #                 },
        #             }
        #         ]
        #     }
        # )

        # return f"data: {chunk}\n\n"
