import re
from typing import List, Optional, Pattern

from adapters.abstract_adapters.openai_sdk_chat_adapter import OpenAISDKChatAdapter
from adapters.abstract_adapters.provider_adapter_mixin import ProviderAdapterMixin
from adapters.types import Model

PROVIDER_NAME = "octoai"
BASE_URL = "https://text.octoai.run/v1"
API_KEY_NAME = "OCTOAI_API_KEY"
API_KEY_PATTERN = re.compile(r".*")


class OctoaiModel(Model):
    provider_name: str = PROVIDER_NAME

    supports_streaming: bool = True
    supports_repeating_roles: bool = True
    supports_system: bool = True
    supports_multiple_system: bool = True
    supports_empty_content: bool = True
    supports_tool_choice_required: bool = True
    supports_last_assistant: bool = True
    supports_first_assistant: bool = True


# TODO: add more models
MODELS: Optional[List[OctoaiModel]] = []


class OctoaiSDKChatProviderAdapter(ProviderAdapterMixin, OpenAISDKChatAdapter):
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
