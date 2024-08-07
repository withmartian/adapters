import re
from typing import Pattern

from adapters.abstract_adapters.openai_sdk_chat_adapter import OpenAISDKChatAdapter
from adapters.abstract_adapters.provider_adapter_mixin import ProviderAdapterMixin
from adapters.types import Model

PROVIDER_NAME = "anyscale"
ANYSCALE_BASE_URL = "https://api.endpoints.anyscale.com/v1"
API_KEY_NAME = "ANYSCALE_API_KEY"
API_KEY_PATTERN = re.compile(r".*")


class AnyscaleModel(Model):
    supports_streaming: bool = True
    supports_multiple_system: bool = False
    provider_name: str = PROVIDER_NAME

    def _get_api_path(self) -> str:
        return f"{self.vendor_name}/{self.name}"


MODELS: list[AnyscaleModel] = []


class AnyscaleSDKChatProviderAdapter(ProviderAdapterMixin, OpenAISDKChatAdapter):
    @staticmethod
    def get_supported_models():
        return MODELS

    @staticmethod
    def get_provider_name() -> str:
        return PROVIDER_NAME

    def get_base_sdk_url(self) -> str:
        return ANYSCALE_BASE_URL

    @staticmethod
    def get_api_key_name() -> str:
        return API_KEY_NAME

    @staticmethod
    def get_api_key_pattern() -> Pattern:
        return API_KEY_PATTERN
