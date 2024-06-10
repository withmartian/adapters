import re
from typing import Optional

from adapters.abstract_adapters.openai_sdk_chat_adapter import OpenAISDKChatAdapter
from adapters.abstract_adapters.provider_adapter_mixin import ProviderAdapterMixin

API_KEY_NAME = "CUSTOM_API_KEY"
API_KEY_PATTERN = re.compile(r"^sk-[a-zA-Z0-9\-]+$")


class CustomAISDKChatProviderAdapter(ProviderAdapterMixin, OpenAISDKChatAdapter):
    def __init__(self, base_url: str):
        self.base_url = base_url
        super().__init__()

    @staticmethod
    def get_api_key_pattern() -> re.Pattern:
        return API_KEY_PATTERN

    def get_custom_sdk_url(self) -> Optional[str]:
        return self.base_url

    @staticmethod
    def get_base_sdk_url() -> str:
        return ""

    @staticmethod
    def get_api_key_name() -> str:
        return API_KEY_NAME
