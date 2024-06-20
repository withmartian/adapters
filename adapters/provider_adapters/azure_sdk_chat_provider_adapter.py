import re
from typing import Pattern

from adapters.abstract_adapters.openai_sdk_chat_adapter import OpenAISDKChatAdapter
from adapters.abstract_adapters.provider_adapter_mixin import ProviderAdapterMixin
from adapters.types import Cost, Model

VENDOR_NAME = "azure"
PROVIDER_NAME = "azure"
BASE_URL = "https://martiantest.openai.azure.com/"
API_KEY_NAME = "AZURE_API_KEY"
API_KEY_PATTERN = re.compile(r".*")


class AzureModel(Model):
    supports_streaming: bool = True
    supports_functions: bool = True
    supports_tools: bool = True
    supports_n: bool = True
    supports_json_output: bool = True
    supports_json_content: bool = True
    vendor_name: str = PROVIDER_NAME
    provider_name: str = PROVIDER_NAME


MODELS = [
    AzureModel(
        name="gpt-35-turbo-16k",
        cost=Cost(prompt=1.0e-6, completion=2.0e-6),
        context_length=16385,
        completion_length=16385,
    )
]


class AzureSDKChatProviderAdapter(ProviderAdapterMixin, OpenAISDKChatAdapter):
    @staticmethod
    def get_supported_models():
        print("heheee")
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
