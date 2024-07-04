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

    def _get_api_path(self) -> str:
        return f"{self.vendor_name}/{self.name}"


MODELS = [
    OpenRouterModel(
        name="dbrx-instruct",
        cost=Cost(prompt=1.08e-6, completion=1.08e-6),
        context_length=32_768,
        vendor_name="databricks",
    ),
    OpenRouterModel(
        name="gemma-7b-it",
        cost=Cost(prompt=0.07e-6, completion=0.07e-6),
        context_length=8192,
        vendor_name="google",
    ),
    OpenRouterModel(
        name="gemma-2-9b-it",
        cost=Cost(prompt=0.2e-6, completion=0.2e-6),
        context_length=8192,
        vendor_name="google",
    ),
    OpenRouterModel(
        name="llama-3-70b-instruct",
        cost=Cost(prompt=0.59e-6, completion=0.59e-6),
        context_length=8192,
        vendor_name="meta-llama",
    ),
    OpenRouterModel(
        name="llama-3-8b-instruct",
        cost=Cost(prompt=0.07e-6, completion=0.07e-6),
        context_length=8192,
        vendor_name="meta-llama",
    ),
    OpenRouterModel(
        name="mistral-7b-instruct",
        cost=Cost(prompt=0.07e-6, completion=0.07e-6),
        context_length=32_768,
        vendor_name="mistralai",
    ),
    OpenRouterModel(
        name="mixtral-8x7b-instruct",
        cost=Cost(prompt=0.24e-6, completion=0.24e-6),
        context_length=32_768,
        vendor_name="mistralai",
    ),
    OpenRouterModel(
        name="mixtral-8x22b-instruct",
        cost=Cost(prompt=0.65e-6, completion=0.65e-6),
        context_length=65_536,
        vendor_name="mistralai",
    ),
    OpenRouterModel(
        name="mythalion-13b",
        cost=Cost(prompt=1.125e-6, completion=1.125e-6),
        context_length=8192,
        vendor_name="pygmalionai",
    ),
    OpenRouterModel(
        name="mythomax-l2-13b",
        cost=Cost(prompt=0.13e-6, completion=0.13e-6),
        context_length=4096,
        vendor_name="gryphe",
    ),
]


class OpenRouterSDKChatProviderAdapter(ProviderAdapterMixin, OpenAISDKChatAdapter):
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
