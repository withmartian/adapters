import re
from typing import Pattern

from adapters.abstract_adapters.openai_sdk_chat_adapter import OpenAISDKChatAdapter
from adapters.abstract_adapters.provider_adapter_mixin import ProviderAdapterMixin
from adapters.types import Cost, Model, ModelPredicates

PROVIDER_NAME = "ai21"
BASE_URL = "https://api.ai21.com/studio/v1"
API_KEY_NAME = "AI21_API_KEY"
API_KEY_PATTERN = re.compile(r".*")
BASE_PREDICATES = ModelPredicates(open_source=True, gdpr_compliant=True)


class AI21Model(Model):
    supports_streaming: bool = True
    supports_empty_content: bool = False
    supports_json_output: bool = True
    supports_json_content: bool = True
    supports_tools: bool = True
    supports_functions: bool = False
    supports_n: bool = True
    provider_name: str = PROVIDER_NAME
    vendor_name: str = PROVIDER_NAME
    predicates: ModelPredicates = BASE_PREDICATES

    def _get_api_path(self) -> str:
        return f"{self.name}"


MODELS: list[AI21Model] = [
    AI21Model(
        name="jamba-1.5-mini",
        cost=Cost(prompt=0.2e-6, completion=0.4e-6),
        context_length=256000,
    ),
    AI21Model(
        name="jamba-1.5-large",
        cost=Cost(prompt=2.0e-6, completion=8.0e-6),
        context_length=256000,
    ),
]


class AI21SDKChatProviderAdapter(ProviderAdapterMixin, OpenAISDKChatAdapter):
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
