from adapters.abstract_adapters.openai_sdk_chat_adapter import OpenAISDKChatAdapter
from adapters.types import Model, ModelProperties, Provider


class OpenRouterModel(Model):
    provider_name: str = Provider.openrouter.value

    properties: ModelProperties = ModelProperties(open_source=True)

    def _get_api_path(self) -> str:
        return f"{self.vendor_name}/{self.name}"


MODELS: list[OpenRouterModel] = []


class OpenRouterSDKChatProviderAdapter(OpenAISDKChatAdapter):
    @staticmethod
    def get_supported_models():
        return MODELS

    @staticmethod
    def get_api_key_name() -> str:
        return "OPENROUTER_API_KEY"

    def get_base_sdk_url(self) -> str:
        return "https://openrouter.ai/api/v1"
