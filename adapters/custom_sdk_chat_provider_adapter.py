from adapters.abstract_adapters.openai_sdk_chat_adapter import OpenAISDKChatAdapter
from adapters.abstract_adapters.provider_adapter_mixin import ProviderAdapterMixin


class CustomAISDKChatProviderAdapter(ProviderAdapterMixin, OpenAISDKChatAdapter):
    def __init__(self, base_url: str):
        self.base_url = base_url
        super().__init__()

    @staticmethod
    def get_supported_models():
        return []

    def get_base_sdk_url(self) -> str:
        return self.base_url

    @staticmethod
    def get_api_key_name() -> str:
        return ""
