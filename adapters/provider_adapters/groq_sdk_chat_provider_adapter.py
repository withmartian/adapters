import re
from typing import Pattern

from adapters.abstract_adapters.openai_sdk_chat_adapter import OpenAISDKChatAdapter
from adapters.abstract_adapters.provider_adapter_mixin import ProviderAdapterMixin
from adapters.types import Cost, Model

PROVIDER_NAME = "groq"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
API_KEY_NAME = "GROQ_API_KEY"
API_KEY_PATTERN = re.compile(r".*")


class GroqModel(Model):
    supports_streaming: bool = True
    provider_name: str = PROVIDER_NAME


MODELS = [
    GroqModel(
        name="mixtral-8x7b-32768",
        cost=Cost(prompt=0.27e-6, completion=0.27e-6),
        context_length=32768,
        vendor_name="mistralai",
    ),
    GroqModel(
        name="llama2-70b-4096",
        cost=Cost(prompt=0.7e-6, completion=0.8e-6),
        context_length=4096,
        vendor_name="meta-llama",
    ),
    GroqModel(
        name="gemma-7b-it",
        cost=Cost(prompt=0.1e-6, completion=0.1e-6),
        context_length=8192,
        vendor_name="google",
    ),
]


class GroqSDKChatProviderAdapter(ProviderAdapterMixin, OpenAISDKChatAdapter):
    @staticmethod
    def get_supported_models():
        return MODELS

    @staticmethod
    def get_provider_name() -> str:
        return PROVIDER_NAME

    @staticmethod
    def get_base_sdk_url() -> str:
        return GROQ_BASE_URL

    @staticmethod
    def get_api_key_name() -> str:
        return API_KEY_NAME

    @staticmethod
    def get_api_key_pattern() -> Pattern:
        return API_KEY_PATTERN

    def get_model_name(self) -> str:
        if self._current_model is None:
            raise ValueError("Model not set")
        return self._current_model.name
