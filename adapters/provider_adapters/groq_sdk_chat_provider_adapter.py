from adapters.abstract_adapters.openai_sdk_chat_adapter import OpenAISDKChatAdapter
from adapters.types import Cost, Model, ModelProperties

PROVIDER_NAME = "groq"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
API_KEY_NAME = "GROQ_API_KEY"
BASE_PROPERTIES = ModelProperties(open_source=True, gdpr_compliant=True)


class GroqModel(Model):
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
        properties=BASE_PROPERTIES.model_copy(update={"gdpr_compliant": False}),
    ),
    GroqModel(
        name="llama3-8b-8192",
        cost=Cost(prompt=0.05e-6, completion=0.08e-6),
        context_length=8192,
        vendor_name="meta-llama",
    ),
    # GroqModel(
    #     name="mixtral-8x7b-32768",
    #     cost=Cost(prompt=0.24e-6, completion=0.24e-6),
    #     context_length=32768,
    #     vendor_name="mistralai",
    # ),
    GroqModel(
        name="gemma-7b-it",
        cost=Cost(prompt=0.07e-6, completion=0.07e-6),
        context_length=8192,
        vendor_name="google",
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
    def get_supported_models():
        return MODELS

    @staticmethod
    def get_provider_name() -> str:
        return PROVIDER_NAME

    def get_base_sdk_url(self) -> str:
        return GROQ_BASE_URL

    @staticmethod
    def get_api_key_name() -> str:
        return API_KEY_NAME
