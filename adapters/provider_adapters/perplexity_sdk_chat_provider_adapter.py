import re
from typing import Pattern

from adapters.abstract_adapters.openai_sdk_chat_adapter import OpenAISDKChatAdapter
from adapters.abstract_adapters.provider_adapter_mixin import ProviderAdapterMixin
from adapters.types import Cost, Model

PROVIDER_NAME = "perplexity"
PERPLEXITY_BASE_URL = "https://api.perplexity.ai"
API_KEY_NAME = "PERPLEXITY_API_KEY"
API_KEY_PATTERN = re.compile(r"^pplx-[a-zA-Z0-9]+$")


class PerplexityModel(Model):
    supports_streaming: bool = True
    vendor_name: str = PROVIDER_NAME
    provider_name: str = PROVIDER_NAME


MODELS = [
    PerplexityModel(
        name="llama-3-70b-instruct",
        cost=Cost(prompt=1.0e-6, completion=1.0e-6),
        context_length=8192,
    ),
    PerplexityModel(
        name="llama-3-8b-instruct",
        cost=Cost(prompt=0.20e-6, completion=0.20e-6),
        context_length=8192,
    ),
    PerplexityModel(
        name="sonar-small-chat",
        cost=Cost(prompt=0.2e-6, completion=0.2e-6),
        context_length=16384,
    ),
    PerplexityModel(
        name="sonar-small-online",
        cost=Cost(prompt=0.2e-6, completion=0.2e-6, request=0.005),
        context_length=12000,
    ),
    PerplexityModel(
        name="sonar-medium-chat",
        cost=Cost(prompt=0.6e-6, completion=0.6e-6),
        context_length=16384,
    ),
    PerplexityModel(
        name="sonar-medium-online",
        cost=Cost(prompt=0.6e-6, completion=0.6e-6, request=0.005),
        context_length=12000,
    ),
    PerplexityModel(
        name="codellama-70b-instruct",
        cost=Cost(prompt=1.0e-6, completion=1.0e-6),
        context_length=16384,
    ),
    PerplexityModel(
        name="mistral-7b-instruct",  # v0.2
        cost=Cost(prompt=0.2e-6, completion=0.2e-6),
        context_length=16384,
    ),
    PerplexityModel(
        name="mixtral-8x7b-instruct",
        cost=Cost(prompt=0.6e-6, completion=0.6e-6),
        context_length=16384,
    ),
]


class PerplexitySDKChatProviderAdapter(ProviderAdapterMixin, OpenAISDKChatAdapter):
    @staticmethod
    def get_supported_models():
        return MODELS

    @staticmethod
    def get_provider_name() -> str:
        return PROVIDER_NAME

    @staticmethod
    def get_base_sdk_url() -> str:
        return PERPLEXITY_BASE_URL

    @staticmethod
    def get_api_key_name() -> str:
        return API_KEY_NAME

    @staticmethod
    def get_api_key_pattern() -> Pattern:
        return API_KEY_PATTERN
