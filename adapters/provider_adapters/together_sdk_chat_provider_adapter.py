from typing import Any, Dict

from adapters.abstract_adapters.openai_sdk_chat_adapter import OpenAISDKChatAdapter
from adapters.types import (
    Cost,
    Model,
    Provider,
    Vendor,
)
from openai.types.chat import ChatCompletionMessageParam


class TogetherModel(Model):
    provider_name: str = Provider.together.value

    def _get_api_path(self) -> str:
        return f"{self.vendor_name}/{self.name}"


MODELS: list[Model] = [
    TogetherModel(
        name="Llama-3-8b-chat-hf",
        cost=Cost(prompt=0.20e-6, completion=0.20e-6),
        context_length=8192,
        vendor_name=Vendor.meta_llama.value,
        supports_json_content=False,
        supports_json_output=False,
        supports_vision=False,
        supports_tools=False,
    ),
    TogetherModel(
        name="Llama-3-70b-chat-hf",
        cost=Cost(prompt=0.90e-6, completion=0.90e-6),
        context_length=8192,
        vendor_name=Vendor.meta_llama.value,
        supports_json_content=False,
        supports_json_output=False,
        supports_vision=False,
        supports_tools=False,
    ),
    TogetherModel(
        name="Meta-Llama-3.1-8B-Instruct-Turbo",
        cost=Cost(prompt=0.18e-6, completion=0.18e-6),
        context_length=131072,
        vendor_name=Vendor.meta_llama.value,
        supports_json_content=False,
        supports_vision=False,
    ),
    TogetherModel(
        name="Meta-Llama-3.1-70B-Instruct-Turbo",
        cost=Cost(prompt=0.88e-6, completion=0.88e-6),
        context_length=131072,
        vendor_name=Vendor.meta_llama.value,
        supports_json_content=False,
        supports_vision=False,
    ),
    TogetherModel(
        name="Meta-Llama-3.1-405B-Instruct-Turbo",
        cost=Cost(prompt=3.5e-6, completion=3.5e-6),
        context_length=130815,
        vendor_name=Vendor.meta_llama.value,
        supports_json_content=False,
        supports_json_output=False,
        supports_vision=False,
    ),
    TogetherModel(
        name="Llama-3.2-3B-Instruct-Turbo",
        cost=Cost(prompt=0.06e-6, completion=0.06e-6),
        context_length=131072,
        vendor_name=Vendor.meta_llama.value,
        supports_json_content=False,
        supports_json_output=False,
        supports_vision=False,
        supports_tools=False,
    ),
    TogetherModel(
        name="Llama-3.2-11B-Vision-Instruct-Turbo",
        cost=Cost(prompt=0.18e-6, completion=0.18e-6),
        context_length=131072,
        vendor_name=Vendor.meta_llama.value,
        supports_json_output=False,
        supports_tools=False,
    ),
    TogetherModel(
        name="Llama-3.2-90B-Vision-Instruct-Turbo",
        cost=Cost(prompt=1.20e-6, completion=1.20e-6),
        context_length=131072,
        vendor_name=Vendor.meta_llama.value,
        supports_json_output=False,
        supports_tools=False,
    ),
    TogetherModel(
        name="Qwen2-72B-Instruct",
        cost=Cost(prompt=0.90e-6, completion=0.90e-6),
        context_length=32768,
        vendor_name=Vendor.qwen.value,
        supports_json_output=False,
        supports_vision=False,
        supports_tools=False,
    ),
    TogetherModel(
        name="Qwen2.5-7B-Instruct-Turbo",
        cost=Cost(prompt=0.30e-6, completion=5.0e-6),
        context_length=32768,
        vendor_name=Vendor.qwen.value,
        supports_json_output=False,
        supports_vision=False,
        supports_tools=False,
    ),
    TogetherModel(
        name="Qwen2.5-72B-Instruct-Turbo",
        cost=Cost(prompt=1.20e-6, completion=1.20e-6),
        context_length=32768,
        vendor_name=Vendor.qwen.value,
        supports_json_output=False,
        supports_vision=False,
        supports_tools=False,
    ),
    TogetherModel(
        name="Mistral-7B-Instruct-v0.3",
        cost=Cost(prompt=0.20e-6, completion=0.20e-6),
        context_length=32768,
        vendor_name=Vendor.mistralai.value,
        supports_json_output=False,
        supports_vision=False,
        supports_tools=False,
        can_system_only=False,
        can_assistant_only=False,
    ),
    TogetherModel(
        name="Mixtral-8x7B-Instruct-v0.1",
        cost=Cost(prompt=0.60e-6, completion=0.60e-6),
        context_length=32768,
        vendor_name=Vendor.mistralai.value,
        supports_vision=False,
        can_system_only=False,
        can_assistant_only=False,
    ),
    TogetherModel(
        name="Mixtral-8x22B-Instruct-v0.1",
        cost=Cost(prompt=1.20e-6, completion=1.20e-6),
        context_length=65536,
        vendor_name=Vendor.mistralai.value,
        supports_json_output=False,
        supports_vision=False,
        supports_tools=False,
        can_system_only=False,
        can_assistant_only=False,
    ),
    TogetherModel(
        name="gemma-2-9b-it",
        cost=Cost(prompt=0.30e-6, completion=0.30e-6),
        context_length=8192,
        vendor_name=Vendor.google.value,
        supports_json_content=False,
        supports_json_output=False,
        supports_vision=False,
        supports_tools=False,
    ),
    TogetherModel(
        name="gemma-2-27b-it",
        cost=Cost(prompt=0.80e-6, completion=0.80e-6),
        context_length=8192,
        vendor_name=Vendor.google.value,
        supports_json_content=False,
        supports_json_output=False,
        supports_vision=False,
        supports_tools=False,
    ),
]

DEFAULT_TEMPERATURE = 0.7


class TogetherSDKChatProviderAdapter(OpenAISDKChatAdapter):
    @staticmethod
    def get_supported_models() -> list[Model]:
        return MODELS

    @staticmethod
    def get_api_key_name() -> str:
        return "TOGETHER_API_KEY"

    def get_base_sdk_url(self) -> str:
        return "https://api.together.xyz"

    def _adjust_temperature(self, temperature: float) -> float:
        return temperature / 2

    def _get_params(
        self, messages: list[ChatCompletionMessageParam], **kwargs: Any
    ) -> Dict[str, Any]:
        params = super()._get_params(messages, **kwargs)

        # If the user has requested n messages, but not specified a temperature, we need to provide default temperature
        if params.get("n") and params.get("temperature") is None:
            params["temperature"] = DEFAULT_TEMPERATURE

        # Keep only last image_url for vision
        skiped_image = False
        for message in reversed(params["messages"]):
            if isinstance(message["content"], list):
                for content in reversed(message["content"]):
                    if content["type"] == "image_url":
                        if skiped_image:
                            content["type"] = "text"
                            content["text"] = content["image_url"]["url"]
                            del content["image_url"]
                        else:
                            skiped_image = True

        return params
