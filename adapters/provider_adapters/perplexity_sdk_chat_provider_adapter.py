from adapters.abstract_adapters.openai_sdk_chat_adapter import OpenAISDKChatAdapter
from adapters.types import Cost, Model, ModelProperties, Provider, Vendor


class PerplexityModel(Model):
    provider_name: str = Provider.perplexity.value


MODELS = [
    PerplexityModel(
        name="llama-3.1-sonar-small-128k-online",
        cost=Cost(prompt=0.2e-6, completion=0.2e-6, request=0.005),
        context_length=127072,
        vendor_name=Vendor.perplexity.value,
    ),
    PerplexityModel(
        name="llama-3.1-sonar-large-128k-online",
        cost=Cost(prompt=1e-6, completion=1e-6, request=0.005),
        context_length=127072,
        vendor_name=Vendor.perplexity.value,
    ),
    PerplexityModel(
        name="llama-3.1-sonar-huge-128k-online",
        cost=Cost(prompt=5.0e-6, completion=5.0e-6, request=0.005),
        context_length=127072,
        vendor_name=Vendor.perplexity.value,
    ),
    PerplexityModel(
        name="llama-3.1-sonar-small-128k-chat",
        cost=Cost(prompt=0.2e-6, completion=0.2e-6),
        context_length=131072,
        vendor_name=Vendor.perplexity.value,
    ),
    PerplexityModel(
        name="llama-3.1-sonar-large-128k-chat",
        cost=Cost(prompt=1.0e-6, completion=1.0e-6),
        context_length=131072,
        vendor_name=Vendor.perplexity.value,
    ),
    PerplexityModel(
        name="llama-3.1-8b-instruct",
        cost=Cost(prompt=0.2e-6, completion=0.2e-6),
        context_length=131072,
        vendor_name=Vendor.meta_llama.value,
        properties=ModelProperties(open_source=True),
    ),
    PerplexityModel(
        name="llama-3.1-70b-instruct",
        cost=Cost(prompt=1e-6, completion=1e-6),
        context_length=131072,
        vendor_name=Vendor.meta_llama.value,
        properties=ModelProperties(open_source=True),
    ),
]


class PerplexitySDKChatProviderAdapter(OpenAISDKChatAdapter):
    @staticmethod
    def get_supported_models():
        return MODELS

    @staticmethod
    def get_api_key_name() -> str:
        return "PERPLEXITY_API_KEY"

    def get_base_sdk_url(self) -> str:
        return "https://api.perplexity.ai"
