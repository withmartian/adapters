from adapters.abstract_adapters.openai_sdk_chat_adapter import OpenAISDKChatAdapter
from adapters.types import Cost, Model, ModelProperties

PROVIDER_NAME = "octoai"
BASE_URL = "https://text.octoai.run/v1"
API_KEY_NAME = "OCTOAI_API_KEY"
BASE_PROPERTIES = ModelProperties(open_source=True)


class OctoaiModel(Model):
    provider_name: str = PROVIDER_NAME
    properties: ModelProperties = BASE_PROPERTIES

    supports_repeating_roles: bool = True
    supports_system: bool = True
    supports_multiple_system: bool = True
    supports_empty_content: bool = True
    supports_tool_choice_required: bool = True
    supports_last_assistant: bool = True
    supports_first_assistant: bool = True
    supports_streaming: bool = True
    supports_temperature: bool = True


MODELS = [
    OctoaiModel(
        name="hermes-2-pro-llama-3-8b",
        cost=Cost(prompt=0.15e-6, completion=0.15e-6),
        context_length=8192,
        vendor_name="hermes-llama",
    ),
    OctoaiModel(
        name="meta-llama-3-70b-instruct",
        cost=Cost(prompt=0.9e-6, completion=0.9e-6),
        context_length=8192,
        vendor_name="meta-llama",
    ),
    OctoaiModel(
        name="nous-hermes-2-mixtral-8x7b-dpo",
        cost=Cost(prompt=0.45e-6, completion=0.45e-6),
        context_length=8192,
        vendor_name="nous-hermes",
    ),
    OctoaiModel(
        name="mixtral-8x7b-instruct",
        cost=Cost(prompt=0.45e-6, completion=0.45e-6),
        context_length=32768,
        vendor_name="mixtral",
    ),
]


class OctoaiSDKChatProviderAdapter(OpenAISDKChatAdapter):
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
