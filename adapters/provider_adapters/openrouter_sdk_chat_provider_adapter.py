import re
from typing import Pattern

from adapters.abstract_adapters.openai_sdk_chat_adapter import OpenAISDKChatAdapter
from adapters.abstract_adapters.provider_adapter_mixin import ProviderAdapterMixin
from adapters.types import Cost, Model

PROVIDER_NAME = "openrouter"
BASE_URL = "https://openrouter.ai/api/v1"
API_KEY_NAME = "OPENROUTER_API_KEY"
API_KEY_PATTERN = re.compile(r".*")


class OpenRouterModel(Model):
    supports_streaming: bool = True
    provider_name: str = PROVIDER_NAME


MODELS = [
    OpenRouterModel(
        name="mythalion-13b",
        cost=Cost(prompt=1.125e-6, completion=1.125e-6),
        context_length=8192,
        completion_length=400,
        vendor_name="pygmalionai",
    ),
    OpenRouterModel(
        name="llama-3-8b-instruct",
        cost=Cost(prompt=0.1e-6, completion=0.1e-6),
        context_length=8192,
        completion_length=400,
        vendor_name="meta-llama",
    ),
]


class OpenRouterSDKChatProviderAdapter(ProviderAdapterMixin, OpenAISDKChatAdapter):
    @staticmethod
    def get_supported_models():
        return MODELS

    @staticmethod
    def get_provider_name() -> str:
        return PROVIDER_NAME

    @staticmethod
    def get_base_sdk_url() -> str:
        return BASE_URL

    @staticmethod
    def get_api_key_name() -> str:
        return API_KEY_NAME

    @staticmethod
    def get_api_key_pattern() -> Pattern:
        return API_KEY_PATTERN
