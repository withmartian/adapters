import re
from typing import Pattern

from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

from adapters.abstract_adapters.openai_sdk_chat_adapter import OpenAISDKChatAdapter
from adapters.abstract_adapters.provider_adapter_mixin import ProviderAdapterMixin
from adapters.types import Cost, Model

PROVIDER_NAME = "fireworks"
BASE_URL = "https://api.fireworks.ai/inference/v1"
API_KEY_NAME = "FIREWORKS_API_KEY"
API_KEY_PATTERN = re.compile(r".*")


class FireworksModel(Model):
    supports_streaming: bool = True
    provider_name: str = PROVIDER_NAME

    def _get_api_path(self) -> str:
        return f"{self.vendor_name}/{self.name}"


MODELS = [
    FireworksModel(
        name="gemma-7b-it",
        cost=Cost(prompt=0.20e-6, completion=0.20e-6),
        context_length=8192,
        vendor_name="accounts/fireworks/models",
        supports_system=False,
        supports_first_assistant=False,
    ),
    FireworksModel(
        name="dbrx-instruct",
        cost=Cost(prompt=1.60e-6, completion=1.60e-6),
        context_length=32_768,
        vendor_name="accounts/fireworks/models",
        supports_system=False,
        supports_first_assistant=False,
    ),
    FireworksModel(
        name="llama-v3-8b-instruct",
        cost=Cost(prompt=0.2e-6, completion=0.2e-6),
        context_length=8192,
        vendor_name="accounts/fireworks/models",
    ),
    FireworksModel(
        name="llama-v3-70b-instruct",
        cost=Cost(prompt=0.90e-6, completion=0.90e-6),
        context_length=8192,
        vendor_name="accounts/fireworks/models",
    ),
    FireworksModel(
        name="mistral-7b-instruct-4k",
        cost=Cost(prompt=0.20e-6, completion=0.20e-6),
        context_length=32_768,
        vendor_name="accounts/fireworks/models",
        supports_first_assistant=False,
    ),
    FireworksModel(
        name="mistral-7b-instruct-v0p2",
        cost=Cost(prompt=0.20e-6, completion=0.20e-6),
        context_length=32_768,
        vendor_name="accounts/fireworks/models",
        supports_first_assistant=False,
    ),
    FireworksModel(
        name="mixtral-8x22b-instruct",
        cost=Cost(prompt=0.90e-6, completion=0.90e-6),
        context_length=65_536,
        vendor_name="accounts/fireworks/models",
    ),
    FireworksModel(
        name="mixtral-8x7b-instruct",
        cost=Cost(prompt=0.50e-6, completion=0.50e-6),
        context_length=32_768,
        vendor_name="accounts/fireworks/models",
        supports_first_assistant=False,
    ),
]


class FireworksSDKChatProviderAdapter(ProviderAdapterMixin, OpenAISDKChatAdapter):
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

    def extract_stream_response(self, request, response: ChatCompletionChunk) -> str:
        if response.choices and response.choices[0].delta.content is None:
            # It must be the first response.
            # Most models start with an empty string.
            response.choices[0].delta.content = ""

        return f"data: {response.model_dump_json()}\n\n"
