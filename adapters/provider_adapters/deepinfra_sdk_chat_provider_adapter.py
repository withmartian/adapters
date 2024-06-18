import re
from typing import Pattern

from adapters.abstract_adapters.openai_sdk_chat_adapter import OpenAISDKChatAdapter
from adapters.abstract_adapters.provider_adapter_mixin import ProviderAdapterMixin
from adapters.types import Cost, Model

PROVIDER_NAME = "deepinfra"
DEEPINFRA_BASE_URL = "https://api.deepinfra.com/v1/openai"
API_KEY_NAME = "DEEPINFRA_API_KEY"
API_KEY_PATTERN = re.compile(r".*")


class DeepInfraModel(Model):
    supports_streaming: bool = True
    provider_name: str = PROVIDER_NAME

    def _get_api_path(self) -> str:
        return f"{self.vendor_name}/{self.name}"


MODELS = [
    DeepInfraModel(
        name="gemma-1.1-7b-it",
        cost=Cost(prompt=0.07e-6, completion=0.07e-6),
        context_length=8192,
        vendor_name="google",
    ),
    DeepInfraModel(
        name="Mistral-7B-Instruct-v0.2",
        cost=Cost(prompt=0.07e-6, completion=0.07e-6),
        context_length=32768,
        vendor_name="mistralai",
    ),
    DeepInfraModel(
        name="Mixtral-8x7B-Instruct-v0.1",
        cost=Cost(prompt=0.24e-6, completion=0.24e-6),
        context_length=32000,
        vendor_name="mistralai",
    ),
    DeepInfraModel(
        name="Mixtral-8x22B-Instruct-v0.1",
        cost=Cost(prompt=0.65e-6, completion=0.65e-6),
        context_length=65536,
        vendor_name="mistralai",
        supports_n=False,
    ),
    DeepInfraModel(
        name="dbrx-instruct",
        cost=Cost(prompt=0.6e-6, completion=0.6e-6),
        context_length=32768,
        vendor_name="databricks",
        supports_n=False,
    ),
    DeepInfraModel(
        name="Meta-Llama-3-70B-Instruct",
        cost=Cost(prompt=0.59e-6, completion=0.79e-6),
        context_length=8000,
        vendor_name="meta-llama",
        supports_n=False,
    ),
    DeepInfraModel(
        name="Meta-Llama-3-8B-Instruct",
        cost=Cost(prompt=0.08e-6, completion=0.08e-6),
        context_length=8000,
        vendor_name="meta-llama",
        supports_n=False,
    ),
]


class DeepInfraSDKChatProviderAdapter(ProviderAdapterMixin, OpenAISDKChatAdapter):
    @staticmethod
    def get_supported_models():
        return MODELS

    @staticmethod
    def get_provider_name() -> str:
        return PROVIDER_NAME

    def get_base_sdk_url(self) -> str:
        return DEEPINFRA_BASE_URL

    @staticmethod
    def get_api_key_name() -> str:
        return API_KEY_NAME

    @staticmethod
    def get_api_key_pattern() -> Pattern:
        return API_KEY_PATTERN
