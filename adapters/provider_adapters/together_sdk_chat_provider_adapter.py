from typing import Any, Dict

from adapters.abstract_adapters.openai_sdk_chat_adapter import OpenAISDKChatAdapter
from adapters.types import Conversation, ConversationRole, Cost, Model, ModelProperties

PROVIDER_NAME = "together"
BASE_URL = "https://api.together.xyz"
API_KEY_NAME = "TOGETHER_API_KEY"
BASE_PROPERTIES = ModelProperties(open_source=True)


class TogetherModel(Model):
    provider_name: str = PROVIDER_NAME
    supports_json_content: bool = True
    supports_streaming: bool = True
    supports_temperature: bool = True

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
]


class TogetherSDKChatProviderAdapter(OpenAISDKChatAdapter):
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

    def _adjust_temperature(self, temperature: float) -> float:
        return temperature / 2

    def _get_params(self, llm_input: Conversation, **kwargs) -> Dict[str, Any]:
        params = super()._get_params(llm_input, **kwargs)
        messages = params["messages"]
        # Remove trailing whitespace from the last assistant message
        if len(messages) > 0 and messages[-1]["role"] == ConversationRole.assistant:
            messages[-1]["content"] = messages[-1]["content"].rstrip()

        return {
            **params,
            "messages": messages,
        }
