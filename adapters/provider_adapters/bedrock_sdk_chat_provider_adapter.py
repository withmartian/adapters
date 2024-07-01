import re
from typing import Pattern

from adapters.abstract_adapters.openai_sdk_chat_adapter import OpenAISDKChatAdapter
from adapters.abstract_adapters.provider_adapter_mixin import ProviderAdapterMixin
from adapters.types import Cost, Model

PROVIDER_NAME = "bedrock"
BASE_URL = ""
API_KEY_NAME = "BEDROCK_API_KEY"
API_KEY_PATTERN = re.compile(r".*")


class BedrockModel(Model):
    supports_streaming: bool = True
    supports_functions: bool = True
    supports_tools: bool = True
    supports_n: bool = True
    supports_json_output: bool = True
    supports_json_content: bool = True
    vendor_name: str = PROVIDER_NAME
    provider_name: str = PROVIDER_NAME


MODELS = [
    BedrockModel(
        name="jamba-instruct",
        cost=Cost(prompt=0.125e-6, completion=0.375e-6),
        context_length=30720,
        completion_length=2048,
    ),
    BedrockModel(
        name="jurassic-2-ultra",
        cost=Cost(prompt=0.125e-6, completion=0.375e-6),
        context_length=30720,
        completion_length=2048,
    ),
    BedrockModel(
        name="jurassic-2-mid",
        cost=Cost(prompt=3.5e-6, completion=10.5e-6),
        context_length=128000,
        completion_length=8192,
    ),
    BedrockModel(
        name="claude-3.5-sonnet",
        cost=Cost(prompt=3.5e-6, completion=10.5e-6),
        context_length=128000,
        completion_length=8192,
    ),
    BedrockModel(
        name="claude-3-opus",
        cost=Cost(prompt=0.35e-6, completion=0.70e-6),
        context_length=128000,
        completion_length=8192,
    ),
    BedrockModel(
        name="claude-3-haiku",
        cost=Cost(prompt=0.35e-6, completion=0.70e-6),
        context_length=128000,
        completion_length=8192,
    ),
    BedrockModel(
        name="claude-3-sonnet",
        cost=Cost(prompt=0.35e-6, completion=0.70e-6),
        context_length=128000,
        completion_length=8192,
    ),
    BedrockModel(
        name="claude-2.1",
        cost=Cost(prompt=0.35e-6, completion=0.70e-6),
        context_length=128000,
        completion_length=8192,
    ),
    BedrockModel(
        name="claude-2.0",
        cost=Cost(prompt=0.35e-6, completion=0.70e-6),
        context_length=128000,
        completion_length=8192,
    ),
    BedrockModel(
        name="claude-instant",
        cost=Cost(prompt=0.35e-6, completion=0.70e-6),
        context_length=128000,
        completion_length=8192,
    ),
]


class BedrockSDKChatProviderAdapter(ProviderAdapterMixin, OpenAISDKChatAdapter):
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
