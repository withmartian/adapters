from adapters.abstract_adapters.openai_sdk_chat_adapter import OpenAISDKChatAdapter
from adapters.types import Cost, Model, ModelProperties, Provider, Vendor


class FireworksModel(Model):
    provider_name: str = Provider.fireworks.value

    properties: ModelProperties = ModelProperties(open_source=True)

    def _get_api_path(self) -> str:
        return f"accounts/fireworks/models/{self.name}"


MODELS = [
    FireworksModel(
        name="llama-v3p1-405b-instruct",
        cost=Cost(prompt=3.00e-6, completion=3.00e-6),
        context_length=131072,
        vendor_name=Vendor.meta_llama.value,
    ),
    FireworksModel(
        name="llama-v3p1-70b-instruct",
        cost=Cost(prompt=0.90e-6, completion=0.90e-6),
        context_length=131072,
        vendor_name=Vendor.meta_llama.value,
    ),
    FireworksModel(
        name="llama-v3p1-8b-instruct",
        cost=Cost(prompt=0.20e-6, completion=0.20e-6),
        context_length=131072,
        vendor_name=Vendor.meta_llama.value,
    ),
    FireworksModel(
        name="llama-v3p2-3b-instruct",
        cost=Cost(prompt=0.10e-6, completion=0.10e-6),
        context_length=131072,
        vendor_name=Vendor.meta_llama.value,
    ),
    FireworksModel(
        name="llama-v3p2-11b-vision-instruct",
        cost=Cost(prompt=0.20e-6, completion=0.20e-6),
        context_length=131072,
        vendor_name=Vendor.meta_llama.value,
    ),
    FireworksModel(
        name="llama-v3p2-1b-instruct",
        cost=Cost(prompt=0.10e-6, completion=0.10e-6),
        context_length=131072,
        vendor_name=Vendor.meta_llama.value,
    ),
    FireworksModel(
        name="llama-v3p2-90b-vision-instruct",
        cost=Cost(prompt=0.90e-6, completion=0.90e-6),
        context_length=131072,
        vendor_name=Vendor.meta_llama.value,
    ),
    FireworksModel(
        name="qwen2p5-72b-instruct",
        cost=Cost(prompt=0.90e-6, completion=0.90e-6),
        context_length=32768,
        vendor_name=Vendor.qwen.value,
    ),
    FireworksModel(
        name="mixtral-8x22b-instruct",
        cost=Cost(prompt=1.20e-6, completion=1.20e-6),
        context_length=65536,
        vendor_name=Vendor.mistralai.value,
    ),
    FireworksModel(
        name="mixtral-8x7b-instruct",
        cost=Cost(prompt=0.50e-6, completion=0.50e-6),
        context_length=32768,
        vendor_name=Vendor.mistralai.value,
    ),
]


class FireworksSDKChatProviderAdapter(OpenAISDKChatAdapter):
    @staticmethod
    def get_api_key_name() -> str:
        return "FIREWORKS_API_KEY"

    @staticmethod
    def get_supported_models():
        return MODELS

    def get_base_sdk_url(self) -> str:
        return "https://api.fireworks.ai/inference/v1"
