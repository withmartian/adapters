import re
from typing import Pattern

from adapters.abstract_adapters.openai_sdk_chat_adapter import OpenAISDKChatAdapter
from adapters.abstract_adapters.provider_adapter_mixin import ProviderAdapterMixin
from adapters.types import Cost, Model

PROVIDER_NAME = "openai"
BASE_URL = "https://api.openai.com/v1"
API_KEY_NAME = "OPENAI_API_KEY"
API_KEY_PATTERN = re.compile(r"^sk-[a-zA-Z0-9]+$")


class OpenAIModel(Model):
    supports_streaming: bool = True
    supports_functions: bool = True
    supports_tools: bool = True
    supports_n: bool = True
    supports_json_output: bool = True
    supports_json_content: bool = True
    vendor_name: str = PROVIDER_NAME
    provider_name: str = PROVIDER_NAME


MODELS = [
    OpenAIModel(
        name="gpt-3.5-turbo-1106",
        cost=Cost(prompt=1.0e-6, completion=2.0e-6),
        context_length=16385,
        completion_length=16385,
    ),
    OpenAIModel(
        name="gpt-3.5-turbo",
        cost=Cost(prompt=0.5e-6, completion=1.5e-6),
        context_length=16385,
        completion_length=16385,
    ),
    OpenAIModel(
        name="gpt-3.5-turbo-0125",
        cost=Cost(prompt=0.5e-6, completion=1.5e-6),
        context_length=16385,
        completion_length=16385,
    ),
    OpenAIModel(
        name="gpt-4-0314",
        cost=Cost(prompt=30.0e-6, completion=60.0e-6),
        context_length=32768,
        completion_length=32768,
        supports_json_output=False,
        supports_functions=False,
        supports_tools=False,
    ),
    OpenAIModel(
        name="gpt-4-0125-preview",
        cost=Cost(prompt=10.0e-6, completion=30.0e-6),
        context_length=128000,
        completion_length=4096,
    ),
    OpenAIModel(
        name="gpt-4-32k-0613",
        cost=Cost(prompt=60.0e-6, completion=120.0e-6),
        context_length=32768,
        completion_length=32768,
        supports_json_output=False,
    ),
    OpenAIModel(
        name="gpt-4-32k",
        cost=Cost(prompt=60.0e-6, completion=120.0e-6),
        context_length=32768,
        completion_length=32768,
        supports_json_output=False,
    ),
    OpenAIModel(
        name="gpt-4-0613",
        cost=Cost(prompt=30.0e-6, completion=60.0e-6),
        context_length=8192,
        completion_length=8192,
        supports_json_output=False,
    ),
    OpenAIModel(
        name="gpt-4",
        cost=Cost(prompt=30.0e-6, completion=60.0e-6),
        context_length=8192,
        completion_length=8192,
        supports_json_output=False,
    ),
    OpenAIModel(
        name="gpt-4-vision-preview",
        cost=Cost(prompt=10.0e-6, completion=30.0e-6),
        context_length=128000,
        completion_length=4096,
        supports_vision=True,
        supports_functions=False,
        supports_tools=False,
        supports_json_output=False,
    ),
    OpenAIModel(
        name="gpt-4-1106-preview",
        cost=Cost(prompt=10.0e-6, completion=30.0e-6),
        context_length=128000,
        completion_length=4096,
    ),
    OpenAIModel(
        name="gpt-4-turbo-preview",
        cost=Cost(prompt=10.0e-6, completion=30.0e-6),
        context_length=128000,
        completion_length=4096,
        supports_json_output=False,
    ),
    OpenAIModel(
        name="gpt-4-0125-preview",
        cost=Cost(prompt=10.0e-6, completion=30.0e-6),
        context_length=128000,
        completion_length=4096,
    ),
    OpenAIModel(
        name="gpt-4-turbo-2024-04-09",
        cost=Cost(prompt=10.0e-6, completion=30.0e-6),
        context_length=128000,
        completion_length=4096,
        supports_vision=True,
    ),
    OpenAIModel(
        name="gpt-4-turbo",
        cost=Cost(prompt=10.0e-6, completion=30.0e-6),
        context_length=128000,
        completion_length=4096,
        supports_vision=True,
    ),
    OpenAIModel(
        name="gpt-4o-2024-05-13",
        cost=Cost(prompt=5.0e-6, completion=15.0e-6),
        context_length=128000,
        completion_length=4096,
        supports_vision=True,
    ),
    OpenAIModel(
        name="gpt-4o",
        cost=Cost(prompt=5.0e-6, completion=15.0e-6),
        context_length=128000,
        completion_length=4096,
        supports_vision=True,
    ),
]


class OpenAISDKChatProviderAdapter(ProviderAdapterMixin, OpenAISDKChatAdapter):
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
