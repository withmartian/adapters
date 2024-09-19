import re
from typing import Pattern

from adapters.abstract_adapters.openai_sdk_chat_adapter import OpenAISDKChatAdapter
from adapters.abstract_adapters.provider_adapter_mixin import ProviderAdapterMixin
from adapters.types import Cost, Model, ModelPredicates

PROVIDER_NAME = "perplexity"
PERPLEXITY_BASE_URL = "https://api.perplexity.ai"
API_KEY_NAME = "PERPLEXITY_API_KEY"
API_KEY_PATTERN = re.compile(r"^pplx-[a-zA-Z0-9]+$")
BASE_PREDICATES = ModelPredicates(open_source=True)


class PerplexityModel(Model):
    supports_streaming: bool = True
    vendor_name: str = PROVIDER_NAME
    provider_name: str = PROVIDER_NAME
    supports_multiple_system: bool = False
    supports_empty_content: bool = False
    supports_first_assistant: bool = False
    supports_last_assistant: bool = False
    predicates: ModelPredicates = BASE_PREDICATES


MODELS = [
    PerplexityModel(
        name="llama-3-sonar-small-32k-chat",
        cost=Cost(prompt=0.2e-6, completion=0.2e-6),
        context_length=32768,
    ),
    PerplexityModel(
        name="llama-3-sonar-small-32k-online",
        cost=Cost(prompt=0.2e-6, completion=0.2e-6, request=0.005),
        context_length=28000,
    ),
    PerplexityModel(
        name="llama-3-sonar-large-32k-chat",
        cost=Cost(prompt=1.0e-6, completion=1.0e-6),
        context_length=32768,
    ),
    PerplexityModel(
        name="llama-3-sonar-large-32k-online",
        cost=Cost(prompt=1.0e-6, completion=1.0e-6, request=0.005),
        context_length=28000,
    ),
    PerplexityModel(
        name="llama-3-8b-instruct",
        cost=Cost(prompt=0.20e-6, completion=0.20e-6),
        context_length=8192,
    ),
    PerplexityModel(
        name="llama-3-70b-instruct",
        cost=Cost(prompt=1.0e-6, completion=1.0e-6),
        context_length=8192,
        predicates=BASE_PREDICATES.model_copy(update={"is_nsfw": True}),
    ),
    PerplexityModel(
        name="mixtral-8x7b-instruct",
        cost=Cost(prompt=0.6e-6, completion=0.6e-6),
        context_length=16384,
    ),
    PerplexityModel(
        name="llama-3.1-sonar-small-128k-online",
        cost=Cost(prompt=0.2e-6, completion=0.2e-6),
        context_length=127072,
        provider_name="perplexity",
    ),
    PerplexityModel(
        name="llama-3.1-sonar-small-128k-chat",
        cost=Cost(prompt=0.2e-6, completion=0.2e-6),
        context_length=131072,
        provider_name="perplexity",
    ),
    PerplexityModel(
        name="llama-3.1-sonar-large-128k-online",
        cost=Cost(prompt=1e-6, completion=1e-6),
        context_length=127072,
        provider_name="perplexity",
    ),
    PerplexityModel(
        name="llama-3.1-sonar-large-128k-chat",
        cost=Cost(prompt=1e-6, completion=1e-6),
        context_length=131072,
        provider_name="perplexity",
    ),
    PerplexityModel(
        name="llama-3.1-8b-instruct",
        cost=Cost(prompt=0.2e-6, completion=0.2e-6),
        context_length=131072,
        provider_name="meta-lama",
    ),
    PerplexityModel(
        name="llama-3.1-70b-instruct",
        cost=Cost(prompt=1e-6, completion=1e-6),
        context_length=131072,
        provider_name="meta-lama",
        predicates=BASE_PREDICATES.model_copy(update={"is_nsfw": True}),
    ),
]


class PerplexitySDKChatProviderAdapter(ProviderAdapterMixin, OpenAISDKChatAdapter):
    @staticmethod
    def get_supported_models():
        return MODELS

    @staticmethod
    def get_provider_name() -> str:
        return PROVIDER_NAME

    def get_base_sdk_url(self) -> str:
        return PERPLEXITY_BASE_URL

    @staticmethod
    def get_api_key_name() -> str:
        return API_KEY_NAME

    @staticmethod
    def get_api_key_pattern() -> Pattern:
        return API_KEY_PATTERN
