from typing import Any
from adapters.abstract_adapters.openai_sdk_chat_adapter import OpenAISDKChatAdapter
from adapters.types import Cost, Model, Provider, Vendor
from openai.types.chat import ChatCompletionMessageParam


class FireworksModel(Model):
    provider_name: str = Provider.fireworks.value

    supports_vision: bool = False

    def _get_api_path(self) -> str:
        if self.name == "yi-large":
            return f"accounts/yi-01-ai/models/{self.name}"

        return f"accounts/fireworks/models/{self.name}"


MODELS: list[Model] = [
    FireworksModel(
        name="llama-v3p1-405b-instruct",
        cost=Cost(prompt=3.00e-6, completion=3.00e-6),
        context_length=131072,
        vendor_name=Vendor.meta_llama.value,
        supports_vision=False,
        supports_tools=False,
    ),
    FireworksModel(
        name="llama-v3p1-70b-instruct",
        cost=Cost(prompt=0.90e-6, completion=0.90e-6),
        context_length=131072,
        vendor_name=Vendor.meta_llama.value,
        supports_vision=False,
        supports_tools=False,
    ),
    FireworksModel(
        name="llama-v3p1-8b-instruct",
        cost=Cost(prompt=0.20e-6, completion=0.20e-6),
        context_length=131072,
        vendor_name=Vendor.meta_llama.value,
        supports_vision=False,
        supports_tools=False,
    ),
    FireworksModel(
        name="llama-v3p2-3b-instruct",
        cost=Cost(prompt=0.20e-6, completion=0.20e-6),
        context_length=131072,
        vendor_name=Vendor.meta_llama.value,
        supports_vision=False,
        supports_tools=False,
    ),
    # FireworksModel(
    #     name="mixtral-8x22b-instruct",
    #     cost=Cost(prompt=0.90e-6, completion=0.90e-6),
    #     context_length=65536,
    #     vendor_name=Vendor.mistralai.value,
    #     supports_vision=False,
    #     supports_tools=False,
    # ),
    FireworksModel(
        name="llama-v3p2-11b-vision-instruct",
        cost=Cost(prompt=0.20e-6, completion=0.20e-6),
        context_length=131072,
        vendor_name=Vendor.meta_llama.value,
        supports_tools=False,
    ),
    FireworksModel(
        name="llama-v3p2-90b-vision-instruct",
        cost=Cost(prompt=0.90e-6, completion=0.90e-6),
        context_length=131072,
        vendor_name=Vendor.meta_llama.value,
        supports_tools=False,
    ),
    # FireworksModel(
    #     name="mixtral-8x7b-instruct-hf",
    #     cost=Cost(prompt=0.50e-6, completion=0.50e-6),
    #     context_length=32768,
    #     vendor_name=Vendor.mistralai.value,
    #     supports_json_content=False,
    #     supports_vision=False,
    #     supports_tools=False,
    #     can_assistant_first=False,
    #     can_repeating_roles=False,
    # ),
    FireworksModel(
        name="yi-large",
        cost=Cost(prompt=3.00e-6, completion=3.00e-6),
        context_length=32768,
        vendor_name=Vendor.O1.value,
        supports_vision=False,
        supports_tools=False,
    ),
    FireworksModel(
        name="llama-v3-70b-instruct-hf",
        cost=Cost(prompt=0.90e-6, completion=0.90e-6),
        context_length=8192,
        vendor_name=Vendor.meta_llama.value,
        supports_vision=False,
        supports_tools=False,
    ),
    FireworksModel(
        name="llama-v3-70b-instruct",
        cost=Cost(prompt=0.90e-6, completion=0.90e-6),
        context_length=8192,
        vendor_name=Vendor.meta_llama.value,
        supports_vision=False,
        supports_tools=False,
    ),
    FireworksModel(
        name="llama-v3-8b-instruct-hf",
        cost=Cost(prompt=0.20e-6, completion=0.20e-6),
        context_length=8192,
        vendor_name=Vendor.meta_llama.value,
        supports_vision=False,
        supports_tools=False,
    ),
    FireworksModel(
        name="llama-v3-8b-instruct",
        cost=Cost(prompt=0.20e-6, completion=0.20e-6),
        context_length=8192,
        vendor_name=Vendor.meta_llama.value,
        supports_vision=False,
        supports_tools=False,
    ),
    FireworksModel(
        name="phi-3-vision-128k-instruct",
        cost=Cost(prompt=0.90e-6, completion=0.90e-6),
        context_length=32064,
        vendor_name=Vendor.microsoft.value,
        supports_tools=False,
        supports_completion=False,
    ),
    # FireworksModel(
    #     name="mixtral-8x7b-instruct",
    #     cost=Cost(prompt=0.50e-6, completion=0.50e-6),
    #     context_length=32768,
    #     vendor_name=Vendor.mistralai.value,
    #     supports_json_content=False,
    #     supports_vision=False,
    #     supports_tools=False,
    #     can_repeating_roles=False,
    #     can_assistant_first=False,
    # ),
    FireworksModel(
        name="mythomax-l2-13b",
        cost=Cost(prompt=0.20e-6, completion=0.20e-6),
        context_length=4096,
        vendor_name=Vendor.gryphe.value,
        supports_vision=False,
        supports_tools=False,
    ),
    FireworksModel(
        name="qwen2p5-72b-instruct",
        cost=Cost(prompt=0.90e-6, completion=0.90e-6),
        context_length=32768,
        vendor_name=Vendor.qwen.value,
        supports_vision=False,
        supports_tools=False,
    ),
    FireworksModel(
        name="llama-v3p2-1b-instruct",
        cost=Cost(prompt=0.20e-6, completion=0.20e-6),
        context_length=131072,
        vendor_name=Vendor.O1.value,
        supports_vision=False,
        supports_tools=False,
    ),
]


class FireworksSDKChatProviderAdapter(OpenAISDKChatAdapter):
    @staticmethod
    def get_supported_models() -> list[Model]:
        return MODELS

    @staticmethod
    def get_api_key_name() -> str:
        return "FIREWORKS_API_KEY"

    def get_base_sdk_url(self) -> str:
        return "https://api.fireworks.ai/inference/v1"

    def _get_params(
        self, messages: list[ChatCompletionMessageParam], **kwargs: Any
    ) -> dict[str, Any]:
        params = super()._get_params(messages, **kwargs)

        # Keep only last image_url for vision
        # skiped_image = False
        # for message in reversed(params["messages"]):
        #     if isinstance(message["content"], list):
        #         for content in reversed(message["content"]):
        #             if content["type"] == "image_url":
        #                 if skiped_image:
        #                     content["type"] = "text"
        #                     content["text"] = content["image_url"]["url"]
        #                     del content["image_url"]
        #                 else:
        #                     skiped_image = True

        # # Remove image details
        # for message in params["messages"]:
        #     if isinstance(message["content"], list):
        #         for content in message["content"]:
        #             if content["type"] == "image_url":
        #                 del content["image_url"]["details"]

        return params
