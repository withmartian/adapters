import re
from typing import Optional, Pattern

from adapters.abstract_adapters.openai_sdk_chat_adapter import OpenAISDKChatAdapter
from adapters.abstract_adapters.provider_adapter_mixin import ProviderAdapterMixin

API_KEY_NAME = ""
API_KEY_PATTERN = re.compile(r".*")


class CustomOpenAISDKChatProviderAdapter(ProviderAdapterMixin, OpenAISDKChatAdapter):
    def __init__(self, base_url: str):
        self.base_url = base_url
        super().__init__()

    def get_custom_sdk_url(self) -> Optional[str]:
        return self.base_url

    @staticmethod
    def get_base_sdk_url() -> str:
        return ""

    @staticmethod
    def get_api_key_name() -> str:
        return API_KEY_NAME

    @staticmethod
    def get_api_key_pattern() -> Pattern:
        return API_KEY_PATTERN
