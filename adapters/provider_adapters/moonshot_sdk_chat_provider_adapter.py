import re
from typing import Pattern

from adapters.abstract_adapters.openai_sdk_chat_adapter import OpenAISDKChatAdapter
from adapters.abstract_adapters.provider_adapter_mixin import ProviderAdapterMixin
from adapters.types import Cost, Model

PROVIDER_NAME = "moonshot"
MOONSHOT_BASE_URL = "https://api.moonshot.cn/v1"
API_KEY_NAME = "MOONSHOT_API_KEY"
API_KEY_PATTERN = re.compile(r".*")


class MoonshotModel(Model):
    supports_streaming: bool = True
    supports_empty_content: bool = False
    provider_name: str = PROVIDER_NAME


# cost measured in CNY, converted to USD on Apr 27 2024
MODELS = [
    MoonshotModel(
        name="moonshot-v1-8k",
        cost=Cost(prompt=1.66e-6, completion=1.66e-6),
        context_length=8000,
        vendor_name="moonshot",
    ),
    MoonshotModel(
        name="moonshot-v1-32k",
        cost=Cost(prompt=3.32e-6, completion=3.32e-6),
        context_length=32000,
        vendor_name="moonshot",
    ),
    MoonshotModel(
        name="moonshot-v1-128k",
        cost=Cost(prompt=8.29e-6, completion=8.29e-6),
        context_length=128000,
        vendor_name="moonshot",
    ),
]


class MoonshotSDKChatProviderAdapter(ProviderAdapterMixin, OpenAISDKChatAdapter):
    @staticmethod
    def get_supported_models():
        return MODELS

    @staticmethod
    def get_provider_name() -> str:
        return PROVIDER_NAME

    def get_base_sdk_url(self) -> str:
        return MOONSHOT_BASE_URL

    @staticmethod
    def get_api_key_name() -> str:
        return API_KEY_NAME

    @staticmethod
    def get_api_key_pattern() -> Pattern:
        return API_KEY_PATTERN
