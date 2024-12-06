from adapters.abstract_adapters.openai_sdk_chat_adapter import OpenAISDKChatAdapter
from adapters.types import Cost, Model, Provider, Vendor


class GroqModel(Model):
    provider_name: str = Provider.groq.value

    supports_vision: bool = False
    supports_tools: bool = False


MODELS: list[Model] = [
    GroqModel(
        name="llama-3.1-70b-versatile",
        cost=Cost(prompt=0.59e-6, completion=0.79e-6),
        context_length=131072,
        vendor_name="meta-llama",
    ),
    GroqModel(
        name="llama-3.1-8b-instant",
        cost=Cost(prompt=0.05e-6, completion=0.08e-6),
        context_length=131072,
        vendor_name="meta-llama",
    ),
    GroqModel(
        name="llama3-70b-8192",
        cost=Cost(prompt=0.59e-6, completion=0.79e-6),
        context_length=8192,
        vendor_name="meta-llama",
    ),
    GroqModel(
        name="llama3-8b-8192",
        cost=Cost(prompt=0.05e-6, completion=0.08e-6),
        context_length=8192,
        vendor_name="meta-llama",
    ),
    GroqModel(
        name="mixtral-8x7b-32768",
        cost=Cost(prompt=0.24e-6, completion=0.24e-6),
        context_length=32768,
        vendor_name="mistralai",
    ),
    GroqModel(
        name="gemma-7b-it",
        cost=Cost(prompt=0.07e-6, completion=0.07e-6),
        context_length=8192,
        vendor_name=Vendor.google.value,
    ),
    GroqModel(
        name="gemma2-9b-it",
        cost=Cost(prompt=0.20e-6, completion=0.20e-6),
        context_length=8192,
        vendor_name="google",
    ),
    GroqModel(
        name="llama3-groq-8b-8192-tool-use-preview",
        cost=Cost(prompt=0.19e-6, completion=0.19e-6),
        context_length=8192,
        vendor_name="groq",
    ),
    GroqModel(
        name="llama-guard-3-8b",
        cost=Cost(prompt=0.20e-6, completion=0.20e-6),
        context_length=8192,
        vendor_name="meta-llama",
    ),
]


class GroqSDKChatProviderAdapter(OpenAISDKChatAdapter):
    @staticmethod
    def get_supported_models() -> list[Model]:
        return MODELS

    @staticmethod
    def get_api_key_name() -> str:
        return "GROQ_API_KEY"

    def get_base_sdk_url(self) -> str:
        return "https://api.groq.com/openai/v1"
