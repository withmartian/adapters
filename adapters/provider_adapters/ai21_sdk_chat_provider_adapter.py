from adapters.abstract_adapters.openai_sdk_chat_adapter import OpenAISDKChatAdapter
from adapters.types import Cost, Model, ModelProperties

PROVIDER_NAME = "ai21"
BASE_URL = "https://api.ai21.com/studio/v1"
API_KEY_NAME = "AI21_API_KEY"
BASE_PROPERTIES = ModelProperties(open_source=True, gdpr_compliant=True)


class AI21Model(Model):
    provider_name: str = PROVIDER_NAME
    vendor_name: str = PROVIDER_NAME
    properties: ModelProperties = BASE_PROPERTIES

    supports_repeating_roles: bool = True
    supports_system: bool = True
    supports_multiple_system: bool = True
    supports_tool_choice_required: bool = True
    supports_last_assistant: bool = True
    supports_first_assistant: bool = True
    supports_streaming: bool = True
    supports_json_output: bool = True
    supports_tools: bool = True
    supports_n: bool = True
    supports_temperature: bool = True

    def _get_api_path(self) -> str:
        return f"{self.name}"


MODELS: list[AI21Model] = [
    AI21Model(
        name="jamba-1.5-mini",
        cost=Cost(prompt=0.2e-6, completion=0.4e-6),
        context_length=256000,
    ),
    AI21Model(
        name="jamba-1.5-large",
        cost=Cost(prompt=2.0e-6, completion=8.0e-6),
        context_length=256000,
    ),
]


class AI21SDKChatProviderAdapter(OpenAISDKChatAdapter):
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
