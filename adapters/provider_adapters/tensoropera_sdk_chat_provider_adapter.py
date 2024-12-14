from adapters.abstract_adapters.openai_sdk_chat_adapter import OpenAISDKChatAdapter
from adapters.types import Cost, Model, Provider, Vendor


class TensorOperaModel(Model):
    provider_name: str = Provider.tensoropera.value
    vendor_name: str = Vendor.moescape.value

    supports_completion: bool = True
    supports_chat: bool = False
    supports_max_completion_tokens: bool = False


MODELS: list[Model] = [
    TensorOperaModel(
        name="euryale",
        api_name="tensoropera-yodayo/L3.1-70B-Euryale-v2.2_gpu4",
        cost=Cost(prompt=0, completion=0),
        context_length=12288,
        can_system_only=False,
    ),
]


class TensorOperaSDKChatProviderAdapter(OpenAISDKChatAdapter):
    @staticmethod
    def get_supported_models() -> list[Model]:
        return MODELS

    @staticmethod
    def get_api_key_name() -> str:
        return "TENSOROPERA_API_KEY"

    def get_base_sdk_url(self) -> str:
        return "https://open.tensoropera.ai/inference/api/v1"
