from adapters.abstract_adapters.openai_sdk_chat_adapter import OpenAISDKChatAdapter
from adapters.types import Cost, Model, Provider, Vendor


class CerebrasModel(Model):
    provider_name: str = Provider.cerebras.value

    supports_vision: bool = False

    def _get_api_path(self) -> str:
        return f"{self.name}"


MODELS: list[CerebrasModel] = [
    CerebrasModel(
        name="llama3.1-8b",
        vendor_name=Vendor.meta_llama.value,
        cost=Cost(prompt=0.1e-6, completion=0.1e-6),
        context_length=128000,
        completion_length=8192,
    ),
    CerebrasModel(
        name="llama3.1-70b",
        vendor_name=Vendor.meta_llama.value,
        cost=Cost(prompt=0.6e-6, completion=0.6e-6),
        context_length=128000,
        completion_length=8192,
    ),
]


class CerebrasSDKChatProviderAdapter(OpenAISDKChatAdapter):
    @staticmethod
    def get_supported_models():
        return MODELS

    @staticmethod
    def get_api_key_name() -> str:
        return "CEREBRAS_API_KEY"

    def get_base_sdk_url(self) -> str:
        return "https://api.cerebras.ai/v1"
