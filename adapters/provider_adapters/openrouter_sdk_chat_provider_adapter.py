import re
from typing import Pattern

from adapters.abstract_adapters.openai_sdk_chat_adapter import OpenAISDKChatAdapter
from adapters.abstract_adapters.provider_adapter_mixin import ProviderAdapterMixin
from adapters.types import Cost, Model, ModelProperties

PROVIDER_NAME = "openrouter"
BASE_URL = "https://openrouter.ai/api/v1"
API_KEY_NAME = "OPENROUTER_API_KEY"
API_KEY_PATTERN = re.compile(r".*")
BASE_PROPERTIES = ModelProperties(open_source=True)


class OpenRouterModel(Model):
    provider_name: str = PROVIDER_NAME
    supports_streaming: bool = True
    supports_repeating_roles: bool = True
    supports_system: bool = True
    supports_multiple_system: bool = True
    supports_empty_content: bool = True
    supports_tool_choice_required: bool = True
    supports_last_assistant: bool = True
    supports_first_assistant: bool = True

    properties: ModelProperties = BASE_PROPERTIES

    def _get_api_path(self) -> str:
        return f"{self.vendor_name}/{self.name}"


# TODO: add more models
MODELS = [
    OpenRouterModel(
        name="dbrx-instruct",
        cost=Cost(prompt=1.08e-6, completion=1.08e-6),
        context_length=32_768,
        vendor_name="databricks",
        properties=BASE_PROPERTIES.model_copy(update={"gdpr_compliant": True}),
    ),
    OpenRouterModel(
        name="mistral-7b-instruct-v2",
        cost=Cost(prompt=0.055e-6, completion=0.055e-6),
        context_length=32_768,
        vendor_name="mistralai",
        properties=BASE_PROPERTIES.model_copy(update={"gdpr_compliant": True}),
    ),
    OpenRouterModel(
        name="mixtral-8x7b-instruct",
        cost=Cost(prompt=0.24e-6, completion=0.24e-6),
        context_length=32_768,
        vendor_name="mistralai",
        properties=BASE_PROPERTIES.model_copy(update={"gdpr_compliant": True}),
    ),
    OpenRouterModel(
        name="mixtral-8x22b-instruct",
        cost=Cost(prompt=0.65e-6, completion=0.65e-6),
        context_length=65_536,
        vendor_name="mistralai",
        properties=BASE_PROPERTIES.model_copy(update={"gdpr_compliant": True}),
    ),
    OpenRouterModel(
        name="mythalion-13b",
        cost=Cost(prompt=1.125e-6, completion=1.125e-6),
        context_length=8192,
        vendor_name="pygmalionai",
    ),
    OpenRouterModel(
        name="qwen-2.5-72b-instruct",
        cost=Cost(prompt=0.35e-6, completion=0.4e-6),
        context_length=131_072,
        vendor_name="qwen",
    ),
    OpenRouterModel(
        name="qwen-2-vl-72b-instruct",
        cost=Cost(prompt=0.4e-6, completion=0.4e-6),
        context_length=32_768,
        vendor_name="qwen",
    ),
    OpenRouterModel(
        name="llama-3.1-lumimaid-8b",
        cost=Cost(prompt=0.1875e-6, completion=1.125e-6),
        context_length=131_072,
        vendor_name="neversleep",
    ),
    OpenRouterModel(
        name="o1-mini-2024-09-12",
        cost=Cost(prompt=3.0e-6, completion=12.0e-6),
        context_length=128_000,
        vendor_name="openai",
    ),
    OpenRouterModel(
        name="o1-mini",
        cost=Cost(prompt=3.0e-6, completion=12.0e-6),
        context_length=128_000,
        vendor_name="openai",
    ),
    OpenRouterModel(
        name="o1-preview-2024-09-12",
        cost=Cost(prompt=15.0e-6, completion=60.0e-6),
        context_length=128_000,
        vendor_name="openai",
    ),
    OpenRouterModel(
        name="o1-preview",
        cost=Cost(prompt=15.0e-6, completion=60.0e-6),
        context_length=128_000,
        vendor_name="openai",
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
