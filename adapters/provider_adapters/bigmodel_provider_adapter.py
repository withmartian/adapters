from adapters.abstract_adapters.openai_sdk_chat_adapter import OpenAISDKChatAdapter
from adapters.general_utils import YUAN_TO_USD
from adapters.types import (
    Cost,
    Model,
    ModelProperties,
    Provider,
    Vendor,
)


class BigModelModel(Model):
    provider_name: str = Provider.bigmodel.value
    vendor_name: str = Vendor.bigmodel.value

    supports_vision: bool = False
    supports_tools: bool = False
    supports_tool_choice: bool = False
    supports_tool_choice_required: bool = False
    supports_n: bool = False
    supports_empty_content: bool = False
    supports_only_system: bool = False
    supports_only_assistant: bool = False

    properties: ModelProperties = ModelProperties(
        gdpr_compliant=False,
    )


MODELS: list[Model] = [
    BigModelModel(
        name="glm-4-plus",
        cost=Cost(prompt=0.05e-6 * YUAN_TO_USD, completion=0.05e-6 * YUAN_TO_USD),
        context_length=128000,
        completion_length=4096,
    ),
    BigModelModel(
        name="glm-4-0520",
        cost=Cost(prompt=0.10e-6 * YUAN_TO_USD, completion=0.10e-6 * YUAN_TO_USD),
        context_length=128000,
        completion_length=4096,
    ),
    BigModelModel(
        name="glm-4-airx",
        cost=Cost(prompt=0.01e-6 * YUAN_TO_USD, completion=0.01e-6 * YUAN_TO_USD),
        context_length=8000,
        completion_length=4096,
    ),
    BigModelModel(
        name="glm-4-air",
        cost=Cost(prompt=0.001e-6 * YUAN_TO_USD, completion=0.001e-6 * YUAN_TO_USD),
        context_length=128000,
        completion_length=4096,
    ),
    BigModelModel(
        name="glm-4-long",
        cost=Cost(prompt=0.001e-6 * YUAN_TO_USD, completion=0.001e-6 * YUAN_TO_USD),
        context_length=1000000,
        completion_length=4096,
    ),
    BigModelModel(
        name="glm-4-flashx",
        cost=Cost(prompt=0.0001e-6 * YUAN_TO_USD, completion=0.0001e-6 * YUAN_TO_USD),
        context_length=128000,
        completion_length=4096,
    ),
    BigModelModel(
        name="glm-4-flash",
        cost=Cost(prompt=0.00e-6 * YUAN_TO_USD, completion=0.00e-6 * YUAN_TO_USD),
        context_length=128000,
        completion_length=4096,
    ),
    BigModelModel(
        name="glm-4v",
        cost=Cost(prompt=0.05 * YUAN_TO_USD, completion=0.05 * YUAN_TO_USD),
        context_length=6000,
        completion_length=4096,
        supports_vision=True,
    ),
    BigModelModel(
        name="glm-4v-plus",
        cost=Cost(prompt=0.01e-6 * YUAN_TO_USD, completion=0.01e-6 * YUAN_TO_USD),
        context_length=6000,
        completion_length=4096,
        supports_vision=True,
    ),
]


class BigModelSDKChatProviderAdapter(OpenAISDKChatAdapter):
    @staticmethod
    def get_supported_models() -> list[Model]:
        return MODELS

    @staticmethod
    def get_api_key_name() -> str:
        return "BIGMODEL_API_KEY"

    def get_base_sdk_url(self) -> str:
        return "https://open.bigmodel.cn/api/paas/v4"
