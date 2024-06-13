import re
from typing import Pattern

from adapters.abstract_adapters.openai_sdk_chat_adapter import OpenAISDKChatAdapter
from adapters.abstract_adapters.provider_adapter_mixin import ProviderAdapterMixin
from adapters.types import Cost, Model

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


MODELS = [
    AnyscaleModel(
        name="gemma-7b-it",
        cost=Cost(prompt=0.15e-6, completion=0.15e-6),
        context_length=8192,
        vendor_name="google",
    ),
    AnyscaleModel(
        name="Llama-2-7b-chat-hf",
        cost=Cost(prompt=0.15e-6, completion=0.15e-6),
        context_length=4096,
        vendor_name="meta-llama",
    ),
    AnyscaleModel(
        name="Llama-2-13b-chat-hf",
        cost=Cost(prompt=0.25e-6, completion=0.25e-6),
        context_length=4096,
        vendor_name="meta-llama",
    ),
    AnyscaleModel(
        name="Llama-2-70b-chat-hf",
        cost=Cost(prompt=1.00e-6, completion=1.00e-6),
        context_length=4096,
        vendor_name="meta-llama",
    ),
    AnyscaleModel(
        name="Meta-Llama-3-70B-Instruct",
        cost=Cost(prompt=1.00e-6, completion=1.00e-6),
        context_length=8192,
        vendor_name="meta-llama",
    ),
    AnyscaleModel(
        name="Meta-Llama-3-8B-Instruct",
        cost=Cost(prompt=0.15e-6, completion=0.15e-6),
        context_length=8192,
        vendor_name="meta-llama",
    ),
    AnyscaleModel(
        name="CodeLlama-34b-Instruct-hf",
        cost=Cost(prompt=1.00e-6, completion=1.00e-6),
        context_length=16384,
        vendor_name="codellama",
    ),
    AnyscaleModel(
        name="CodeLlama-70b-Instruct-hf",
        cost=Cost(prompt=1.00e-6, completion=1.00e-6),
        context_length=16384,
        vendor_name="codellama",
    ),
    # AnyscaleModel(
    #     name="zephyr-7b-beta",
    #     cost=Cost(prompt=0.15e-6, completion=0.15e-6),
    #     context_length=16384,
    #     vendor_name="HuggingFaceH4",
    # ),
    AnyscaleModel(
        name="Mistral-7B-Instruct-v0.1",
        cost=Cost(prompt=0.15e-6, completion=0.15e-6),
        context_length=16384,
        vendor_name="mistralai",
    ),
    AnyscaleModel(
        name="Mixtral-8x7B-Instruct-v0.1",
        cost=Cost(prompt=0.50e-6, completion=0.50e-6),
        context_length=32768,
        vendor_name="mistralai",
    ),
    AnyscaleModel(
        name="Mixtral-8x22B-Instruct-v0.1",
        cost=Cost(prompt=0.9e-6, completion=0.90e-6),
        context_length=65536,
        vendor_name="mistralai",
    ),
    AnyscaleModel(
        name="NeuralHermes-2.5-Mistral-7B",
        cost=Cost(prompt=0.15e-6, completion=0.15e-6),
        context_length=16384,
        vendor_name="mlabonne",
    ),
    AnyscaleModel(
        name="Llama-3-8b-chat-hf",
        cost=Cost(prompt=0.15e-6, completion=0.15e-6),
        context_length=8000,
        vendor_name="meta-llama",
    ),
    AnyscaleModel(
        name="Llama-3-70b-chat-hf",
        cost=Cost(prompt=1.0e-6, completion=1.0e-6),
        context_length=8000,
        vendor_name="meta-llama",
    ),
]


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
