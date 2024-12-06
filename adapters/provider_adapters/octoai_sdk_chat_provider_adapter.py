from adapters.abstract_adapters.openai_sdk_chat_adapter import OpenAISDKChatAdapter
from adapters.types import Cost, Model, Provider, Vendor


class OctoaiModel(Model):
    provider_name: str = Provider.octoai.value

    supports_vision: bool = False
    supports_tools: bool = False


MODELS: list[Model] = [
    OctoaiModel(
        name="hermes-2-pro-llama-3-8b",
        cost=Cost(prompt=0.15e-6, completion=0.15e-6),
        context_length=8192,
        vendor_name=Vendor.hermes_llama.value,
    ),
    OctoaiModel(
        name="meta-llama-3-70b-instruct",
        cost=Cost(prompt=0.9e-6, completion=0.9e-6),
        context_length=8192,
        vendor_name=Vendor.meta_llama.value,
    ),
    OctoaiModel(
        name="nous-hermes-2-mixtral-8x7b-dpo",
        cost=Cost(prompt=0.45e-6, completion=0.45e-6),
        context_length=8192,
        vendor_name=Vendor.nous_hermes.value,
    ),
    OctoaiModel(
        name="mixtral-8x7b-instruct",
        cost=Cost(prompt=0.45e-6, completion=0.45e-6),
        context_length=32768,
        vendor_name=Vendor.mixtral.value,
    ),
]


class OctoaiSDKChatProviderAdapter(OpenAISDKChatAdapter):
    @staticmethod
    def get_supported_models() -> list[Model]:
        return MODELS

    @staticmethod
    def get_api_key_name() -> str:
        return "OCTOAI_API_KEY"

    def get_base_sdk_url(self) -> str:
        return "https://text.octoai.run/v1"
