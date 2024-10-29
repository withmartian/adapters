from adapters.abstract_adapters.openai_sdk_chat_adapter import OpenAISDKChatAdapter
from adapters.types import Cost, Model, ModelProperties

PROVIDER_NAME = "perplexity"
PERPLEXITY_BASE_URL = "https://api.perplexity.ai"
API_KEY_NAME = "PERPLEXITY_API_KEY"
BASE_PROPERTIES = ModelProperties(open_source=True)


class PerplexityModel(Model):
    vendor_name: str = PROVIDER_NAME
    provider_name: str = PROVIDER_NAME

    supports_repeating_roles: bool = True
    supports_system: bool = True
    supports_tool_choice_required: bool = True
    supports_streaming: bool = True
    supports_temperature: bool = True

    properties: ModelProperties = BASE_PROPERTIES


MODELS = [
    PerplexityModel(
        name="llama-3.1-sonar-small-128k-online",
        cost=Cost(prompt=0.2e-6, completion=0.2e-6, request=0.005),
        context_length=127072,
    ),
    PerplexityModel(
        name="llama-3.1-sonar-large-128k-online",
        cost=Cost(prompt=1e-6, completion=1e-6, request=0.005),
        context_length=127072,
    ),
    PerplexityModel(
        name="llama-3.1-sonar-huge-128k-online",
        cost=Cost(prompt=5.0e-6, completion=5.0e-6, request=0.005),
        context_length=127072,
    ),
    PerplexityModel(
        name="llama-3.1-sonar-small-128k-chat",
        cost=Cost(prompt=0.2e-6, completion=0.2e-6),
        context_length=131072,
    ),
    PerplexityModel(
        name="llama-3.1-sonar-large-128k-chat",
        cost=Cost(prompt=1e-6, completion=1e-6),
        context_length=131072,
    ),
    # PerplexityModel(
    #     name="llama-3.1-sonar-huge-128k-chat",
    #     cost=Cost(prompt=5.0e-6, completion=5.0e-6),
    #     context_length=131072,
    # ),
    PerplexityModel(
        name="llama-3.1-8b-instruct",
        cost=Cost(prompt=0.2e-6, completion=0.2e-6),
        context_length=131072,
        vendor_name="meta-llama",
    ),
    PerplexityModel(
        name="llama-3.1-70b-instruct",
        cost=Cost(prompt=1e-6, completion=1e-6),
        context_length=131072,
        vendor_name="meta-llama",
        properties=BASE_PROPERTIES.model_copy(update={"is_nsfw": True}),
    ),
]


class PerplexitySDKChatProviderAdapter(OpenAISDKChatAdapter):
    @staticmethod
    def get_supported_models():
        return MODELS

    @staticmethod
    def get_provider_name() -> str:
        return PROVIDER_NAME

    def get_base_sdk_url(self) -> str:
        return PERPLEXITY_BASE_URL

    @staticmethod
    def get_api_key_name() -> str:
        return API_KEY_NAME
