from typing import Any, Dict

from adapters.abstract_adapters.openai_sdk_chat_adapter import OpenAISDKChatAdapter
from adapters.types import (
    Conversation,
    ConversationRole,
    Cost,
    Model,
    Provider,
    Vendor,
)


class TogetherModel(Model):
    provider_name: str = Provider.together.value

    supports_user: bool = True
    supports_repeating_roles: bool = True
    supports_streaming: bool = True
    supports_tools: bool = True
    supports_n: bool = False  # Suports with temperature
    supports_system: bool = True
    supports_multiple_system: bool = True
    supports_empty_content: bool = True
    supports_tool_choice_required: bool = True
    supports_json_output: bool = True
    supports_json_content: bool = True
    supports_last_assistant: bool = True
    supports_first_assistant: bool = True
    supports_temperature: bool = True

    def _get_api_path(self) -> str:
        return f"{self.vendor_name}/{self.name}"


MODELS = [
    TogetherModel(
        name="Meta-Llama-3.1-8B-Instruct-Turbo",
        cost=Cost(prompt=0.18e-6, completion=0.18e-6),
        context_length=8192,
        vendor_name=Vendor.meta_llama.value,
    ),
    TogetherModel(
        name="Meta-Llama-3.1-70B-Instruct-Turbo",
        cost=Cost(prompt=0.88e-6, completion=0.88e-6),
        context_length=8192,
        vendor_name=Vendor.meta_llama.value,
    ),
    TogetherModel(
        name="Meta-Llama-3.1-405B-Instruct-Turbo",
        cost=Cost(prompt=5.0e-6, completion=5.0e-6),
        context_length=8192,
        vendor_name=Vendor.meta_llama.value,
        supports_json_output=False,
    ),
]


class TogetherSDKChatProviderAdapter(OpenAISDKChatAdapter):
    @staticmethod
    def get_supported_models():
        return MODELS

    @staticmethod
    def get_provider_name() -> str:
        return Provider.together.value

    def get_base_sdk_url(self) -> str:
        return "https://api.together.xyz"

    @staticmethod
    def get_api_key_name() -> str:
        return "TOGETHER_API_KEY"

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
