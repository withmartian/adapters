from adapters.abstract_adapters.openai_sdk_chat_adapter import OpenAISDKChatAdapter
from adapters.types import Model, ModelProperties, Provider, Vendor


class MoonshotModel(Model):
    provider_name: str = Provider.moonshot.value
    vendor_name: str = Vendor.moonshot.value

    supports_completion: bool = False
    supports_vision: bool = False
    supports_tools: bool = False
    supports_n: bool = False
    supports_max_completion_tokens: bool = False
    supports_max_tokens: bool = False

    can_empty_content: bool = False
    can_min_p: bool = False
    can_top_p: bool = False

    properties: ModelProperties = ModelProperties(chinese=True)


# Cost measured in CNY, converted to USD on Apr 27 2024
MODELS: list[Model] = [
    # MoonshotModel(
    #     name="moonshot-v1-8k",
    #     cost=Cost(prompt=1.66e-6, completion=1.66e-6),
    #     context_length=8000,
    # ),
    # MoonshotModel(
    #     name="moonshot-v1-32k",
    #     cost=Cost(prompt=3.32e-6, completion=3.32e-6),
    #     context_length=32000,
    # ),
    # MoonshotModel(
    #     name="moonshot-v1-128k",
    #     cost=Cost(prompt=8.29e-6, completion=8.29e-6),
    #     context_length=128000,
    # ),
]


class MoonshotSDKChatProviderAdapter(OpenAISDKChatAdapter):
    @staticmethod
    def get_supported_models() -> list[Model]:
        return MODELS

    @staticmethod
    def get_api_key_name() -> str:
        return "MOONSHOT_API_KEY"

    def get_base_sdk_url(self) -> str:
        return "https://api.moonshot.cn/v1"
