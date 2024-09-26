import re
from typing import Any, Dict, Pattern

from adapters.abstract_adapters.openai_sdk_chat_adapter import OpenAISDKChatAdapter
from adapters.abstract_adapters.provider_adapter_mixin import ProviderAdapterMixin
from adapters.types import Conversation, ConversationRole, Cost, Model, ModelProperties

PROVIDER_NAME = "together"
BASE_URL = "https://api.together.xyz"
API_KEY_NAME = "TOGETHER_API_KEY"
API_KEY_PATTERN = re.compile(r".*")
BASE_PROPERTIES = ModelProperties(open_source=True)


class TogetherModel(Model):
    supports_streaming: bool = True
    supports_json_content: bool = True
    provider_name: str = PROVIDER_NAME
    properties: ModelProperties = BASE_PROPERTIES

    def _get_api_path(self) -> str:
        return f"{self.vendor_name}/{self.name}"


MODELS = [
    TogetherModel(
        name="Meta-Llama-3.1-8B-Instruct-Turbo",
        cost=Cost(prompt=0.18e-6, completion=0.18e-6),
        context_length=8192,
        vendor_name="meta-llama",
        supports_json_content=False,
    ),
    TogetherModel(
        name="Meta-Llama-3.1-70B-Instruct-Turbo",
        cost=Cost(prompt=0.88e-6, completion=0.88e-6),
        context_length=8192,
        vendor_name="meta-llama",
        supports_json_content=False,
        properties=BASE_PROPERTIES.model_copy(update={"is_nsfw": True}),
    ),
    TogetherModel(
        name="Meta-Llama-3.1-405B-Instruct-Turbo",
        cost=Cost(prompt=5.0e-6, completion=5.0e-6),
        context_length=8192,
        vendor_name="meta-llama",
        supports_json_content=False,
    ),
    TogetherModel(
        name="Meta-Llama-3-8B-Instruct-Turbo",
        cost=Cost(prompt=0.18e-6, completion=0.18e-6),
        context_length=8192,
        vendor_name="meta-llama",
        supports_json_content=False,
    ),
    TogetherModel(
        name="Meta-Llama-3-70B-Instruct-Turbo",
        cost=Cost(prompt=0.88e-6, completion=0.88e-6),
        context_length=8192,
        vendor_name="meta-llama",
        supports_json_content=False,
        properties=BASE_PROPERTIES.model_copy(update={"is_nsfw": True}),
    ),
    TogetherModel(
        name="Meta-Llama-3-8B-Instruct-Lite",
        cost=Cost(prompt=0.1e-6, completion=0.1e-6),
        context_length=8192,
        vendor_name="meta-llama",
        supports_json_content=False,
    ),
    TogetherModel(
        name="Meta-Llama-3-70B-Instruct-Lite",
        cost=Cost(prompt=0.54e-6, completion=0.54e-6),
        context_length=8192,
        vendor_name="meta-llama",
        supports_json_content=False,
        properties=BASE_PROPERTIES.model_copy(update={"is_nsfw": True}),
    ),
    TogetherModel(
        name="Llama-2-13b-chat-hf",
        cost=Cost(prompt=0.22e-6, completion=0.22e-6),
        context_length=4096,
        vendor_name="meta-llama",
    ),
    TogetherModel(
        name="Llama-3-8b-chat-hf",
        cost=Cost(prompt=0.2e-6, completion=0.2e-6),
        context_length=8192,
        vendor_name="meta-llama",
        supports_json_content=False,
    ),
    TogetherModel(
        name="Llama-3-70b-chat-hf",
        cost=Cost(prompt=0.9e-6, completion=0.9e-6),
        context_length=8000,
        vendor_name="meta-llama",
        supports_json_content=False,
        properties=BASE_PROPERTIES.model_copy(update={"is_nsfw": True}),
    ),
]


class TogetherSDKChatProviderAdapter(ProviderAdapterMixin, OpenAISDKChatAdapter):
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

    def adjust_temperature(self, temperature: float) -> float:
        return temperature / 2

    def get_params(self, llm_input: Conversation, **kwargs) -> Dict[str, Any]:
        params = super().get_params(llm_input, **kwargs)
        messages = params["messages"]
        # Remove trailing whitespace from the last assistant message
        if len(messages) > 0 and messages[-1]["role"] == ConversationRole.assistant:
            messages[-1]["content"] = messages[-1]["content"].rstrip()

        return {
            **params,
            "messages": messages,
        }
