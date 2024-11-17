from adapters.abstract_adapters.openai_sdk_chat_adapter import OpenAISDKChatAdapter
from adapters.types import Cost, Model, ModelProperties, Provider, Vendor


class BigModelModel(Model):
    provider_name: str = Provider.bigmodel.value
    vendor_name: str = Vendor.bigmodel.value

    properties: ModelProperties = ModelProperties(gdpr_compliant=False)


MODELS: list[Model] = [
    BigModelModel(
        name="glm-4",
        cost=Cost(
            prompt=0.0, completion=0.0
        ),  # Update with actual costs when available
        context_length=128000,  # Based on documentation
        completion_length=4096,
        supports_vision=False,
    ),
    BigModelModel(
        name="glm-4-plus",
        cost=Cost(
            prompt=0.0, completion=0.0
        ),  # Update with actual costs when available
        context_length=128000,
        completion_length=4096,
        supports_vision=False,
    ),
    BigModelModel(
        name="glm-4v",
        cost=Cost(
            prompt=0.0, completion=0.0
        ),  # Update with actual costs when available
        context_length=6000,  # Based on image size limit documentation
        completion_length=4096,
        supports_vision=True,
    ),
    BigModelModel(
        name="glm-4v-plus",
        cost=Cost(
            prompt=0.0, completion=0.0
        ),  # Update with actual costs when available
        context_length=6000,  # Based on image size limit documentation
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
