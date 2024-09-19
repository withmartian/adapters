import re
from typing import Pattern

from adapters.abstract_adapters.openai_sdk_chat_adapter import OpenAISDKChatAdapter
from adapters.abstract_adapters.provider_adapter_mixin import ProviderAdapterMixin
from adapters.types import Cost, Model, ModelPredicates

PROVIDER_NAME = "openrouter"
BASE_URL = "https://openrouter.ai/api/v1"
API_KEY_NAME = "OPENROUTER_API_KEY"
API_KEY_PATTERN = re.compile(r".*")
BASE_PREDICATES = ModelPredicates(open_source=True)


class OpenRouterModel(Model):
    supports_streaming: bool = True
    provider_name: str = PROVIDER_NAME
    predicates: ModelPredicates = BASE_PREDICATES

    def _get_api_path(self) -> str:
        return f"{self.vendor_name}/{self.name}"


MODELS = [
    OpenRouterModel(
        name="dbrx-instruct",
        cost=Cost(prompt=1.08e-6, completion=1.08e-6),
        context_length=32_768,
        vendor_name="databricks",
        predicates=BASE_PREDICATES.model_copy(update={"gdpr_compliant": True}),
    ),
    OpenRouterModel(
        name="gemma-7b-it",
        cost=Cost(prompt=0.07e-6, completion=0.07e-6),
        context_length=8192,
        vendor_name="google",
        predicates=BASE_PREDICATES.model_copy(update={"gdpr_compliant": True}),
    ),
    OpenRouterModel(
        name="gemma-2-9b-it",
        cost=Cost(prompt=0.06e-6, completion=0.06e-6),
        context_length=8192,
        vendor_name="google",
        predicates=BASE_PREDICATES.model_copy(update={"gdpr_compliant": True}),
    ),
    OpenRouterModel(
        name="llama-3-70b-instruct",
        cost=Cost(prompt=0.35e-6, completion=0.35e-6),
        context_length=8192,
        vendor_name="meta-llama",
        predicates=BASE_PREDICATES.model_copy(update={"is_nsfw": True}),
    ),
    OpenRouterModel(
        name="llama-3-8b-instruct",
        cost=Cost(prompt=0.055e-6, completion=0.55e-6),
        context_length=8192,
        vendor_name="meta-llama",
    ),
    OpenRouterModel(
        name="mistral-7b-instruct-v2",
        cost=Cost(prompt=0.055e-6, completion=0.055e-6),
        context_length=32_768,
        vendor_name="mistralai",
        predicates=BASE_PREDICATES.model_copy(update={"gdpr_compliant": True}),
    ),
    OpenRouterModel(
        name="mixtral-8x7b-instruct",
        cost=Cost(prompt=0.24e-6, completion=0.24e-6),
        context_length=32_768,
        vendor_name="mistralai",
        predicates=BASE_PREDICATES.model_copy(update={"gdpr_compliant": True}),
    ),
    OpenRouterModel(
        name="mixtral-8x22b-instruct",
        cost=Cost(prompt=0.65e-6, completion=0.65e-6),
        context_length=65_536,
        vendor_name="mistralai",
        predicates=BASE_PREDICATES.model_copy(update={"gdpr_compliant": True}),
    ),
    OpenRouterModel(
        name="mythalion-13b",
        cost=Cost(prompt=1.125e-6, completion=1.125e-6),
        context_length=8192,
        vendor_name="pygmalionai",
    ),
    OpenRouterModel(
        name="mythomax-l2-13b",
        cost=Cost(prompt=0.1e-6, completion=0.1e-6),
        context_length=4096,
        vendor_name="gryphe",
    ),
    OpenRouterModel(
        name="llama-3.1-sonar-large-128k-online",
        cost=Cost(prompt=1.0e-6, completion=1.0e-6),
        context_length=131072,
        vendor_name="perplexity",
        supports_empty_content=False,
    ),
    OpenRouterModel(
        name="llama-3.1-sonar-small-128k-chat",
        cost=Cost(prompt=0.2e-6, completion=0.2e-6),
        context_length=131072,
        vendor_name="perplexity",
        supports_empty_content=False,
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
