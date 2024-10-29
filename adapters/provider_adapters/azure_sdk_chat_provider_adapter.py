from typing import Any, Callable, Dict

from openai import AsyncAzureOpenAI, AzureOpenAI, OpenAI
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    Choice as ChoiceChunk,
    ChoiceDelta,
)

from adapters.abstract_adapters.openai_sdk_chat_adapter import OpenAISDKChatAdapter
from adapters.types import (
    AdapterChatCompletionChunk,
    Conversation,
    ConversationRole,
    Cost,
    Model,
    ModelProperties,
)

VENDOR_NAME = "openai"
PROVIDER_NAME = "azure"
BASE_URL = "https://martiantest.openai.azure.com/"
API_KEY_NAME = "AZURE_API_KEY"
BASE_PROPERTIES = ModelProperties(gdpr_compliant=True)


class AzureModel(Model):
    vendor_name: str = VENDOR_NAME
    provider_name: str = PROVIDER_NAME

    supports_repeating_roles: bool = True
    supports_system: bool = True
    supports_multiple_system: bool = True
    supports_empty_content: bool = True
    supports_last_assistant: bool = True
    supports_first_assistant: bool = True
    supports_functions: bool = True
    supports_tools: bool = True
    supports_n: bool = True
    supports_json_output: bool = True
    supports_json_content: bool = True
    supports_streaming: bool = True
    supports_temperature: bool = True

    properties: ModelProperties = BASE_PROPERTIES


MODELS = [
    AzureModel(
        name="gpt-4o",
        cost=Cost(prompt=5.0e-6, completion=15.0e-6),
        context_length=128000,
        completion_length=4096,
    ),
    AzureModel(
        name="gpt-4o-mini",
        cost=Cost(prompt=0.15e-6, completion=0.6e-6),
        context_length=128000,
        completion_length=16385,
    ),
]


class AzureSDKChatProviderAdapter(OpenAISDKChatAdapter):
    @staticmethod
    def get_supported_models():
        return MODELS

    @staticmethod
    def get_provider_name() -> str:
        return PROVIDER_NAME

    @staticmethod
    def get_api_key_name() -> str:
        return API_KEY_NAME

    def _call_sync(self) -> Callable:
        return self._client_sync.chat.completions.create

    def _call_async(self) -> Callable:
        return self._client_async.chat.completions.create

    def _create_client_sync(self, base_url: str, api_key: str) -> OpenAI:
        return AzureOpenAI(
            api_key=api_key,
            azure_endpoint=base_url,
            api_version="2024-06-01",
        )

    def _create_client_async(self, base_url: str, api_key: str) -> AsyncAzureOpenAI:
        return AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=base_url,
            api_version="2024-06-01",
        )

    def get_base_sdk_url(self) -> str:
        return BASE_URL

    def _get_params(self, llm_input: Conversation, **kwargs) -> Dict[str, Any]:
        params = super()._get_params(llm_input, **kwargs)

        azure_tool_choice = kwargs.get("tool_choice")

        if azure_tool_choice == "required":
            azure_tool_choice = "auto"

        return {
            **params,
            "tool_choice": azure_tool_choice,
        }

    def _extract_stream_response(
        self, request, response: ChatCompletionChunk, state: dict
    ) -> AdapterChatCompletionChunk:
        adapter_response = AdapterChatCompletionChunk.model_construct(
            **response.model_dump(),
        )

        if len(adapter_response.choices) == 0:
            adapter_response.choices = [
                ChoiceChunk(
                    index=0,
                    delta=ChoiceDelta(
                        role=ConversationRole.assistant.value, content=""
                    ),
                )
            ]

        return adapter_response
