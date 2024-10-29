from adapters.abstract_adapters.openai_sdk_chat_adapter import OpenAISDKChatAdapter
from adapters.types import Cost, Model

PROVIDER_NAME = "cerebras"
BASE_URL = "https://api.cerebras.ai/v1"
API_KEY_NAME = "CEREBRAS_API_KEY"


class CerebrasModel(Model):
    provider_name: str = PROVIDER_NAME

    supports_user: bool = True
    supports_repeating_roles: bool = True
    supports_streaming: bool = True
    supports_tools: bool = True
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
        return f"{self.name}"


MODELS: list[CerebrasModel] = [
    CerebrasModel(
        name="llama3.1-8b",
        vendor_name="meta-llama",
        cost=Cost(prompt=0.1e-6, completion=0.1e-6),
        context_length=128000,
        completion_length=8192,
    ),
    CerebrasModel(
        name="llama3.1-70b",
        vendor_name="meta-llama",
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
    def get_provider_name() -> str:
        return PROVIDER_NAME

    def get_base_sdk_url(self) -> str:
        return BASE_URL

    @staticmethod
    def get_api_key_name() -> str:
        return API_KEY_NAME
