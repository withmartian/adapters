import re
from typing import Any, Dict, Pattern

from openai import AsyncAzureOpenAI, AzureOpenAI

from adapters.abstract_adapters.openai_sdk_chat_adapter import OpenAISDKChatAdapter
from adapters.abstract_adapters.provider_adapter_mixin import ProviderAdapterMixin
from adapters.types import Conversation, Cost, Model, ModelProperties

VENDOR_NAME = "openai"
PROVIDER_NAME = "azure"
BASE_URL = "https://martiantest.openai.azure.com/"
API_KEY_NAME = "AZURE_API_KEY"
API_KEY_PATTERN = re.compile(r".*")
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
    supports_streaming: bool = True
    supports_functions: bool = True
    supports_tools: bool = True
    supports_n: bool = True
    supports_json_output: bool = True
    supports_json_content: bool = True

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


class AzureSDKChatProviderAdapter(ProviderAdapterMixin, OpenAISDKChatAdapter):
    _sync_client: AzureOpenAI
    _async_client: AsyncAzureOpenAI

    def __init__(
        self,
    ):
        super().__init__()
        self._sync_client = AzureOpenAI(
            api_key=self.get_api_key(),
            azure_endpoint=self.get_base_sdk_url(),
            api_version="2024-05-01-preview",
        )
        self._async_client = AsyncAzureOpenAI(
            api_key=self.get_api_key(),
            azure_endpoint=self.get_base_sdk_url(),
            api_version="2024-05-01-preview",
        )

    @staticmethod
    def get_supported_models():
        return MODELS

    @staticmethod
    def get_provider_name() -> str:
        return PROVIDER_NAME

    def get_base_sdk_url(self) -> str:
        return BASE_URL

    @staticmethod
    def get_api_key_name() -> str:
        return API_KEY_NAME

    @staticmethod
    def get_api_key_pattern() -> Pattern:
        return API_KEY_PATTERN

    def get_params(self, llm_input: Conversation, **kwargs) -> Dict[str, Any]:
        params = super().get_params(llm_input, **kwargs)

        azure_tool_choice = kwargs.get("tool_choice")

        if azure_tool_choice == "required":
            azure_tool_choice = "auto"

        return {
            **params,
            "tool_choice": azure_tool_choice,
        }
