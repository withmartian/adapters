from adapters.abstract_adapters.openai_sdk_chat_adapter import OpenAISDKChatAdapter
from adapters.types import Cost, Model, ModelProperties

PROVIDER_NAME = "deepinfra"
DEEPINFRA_BASE_URL = "https://api.deepinfra.com/v1/openai"
API_KEY_NAME = "DEEPINFRA_API_KEY"
BASE_PROPERTIES = ModelProperties(open_source=True, gdpr_compliant=True)


class DeepInfraModel(Model):
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

    def _get_api_path(self) -> str:
        return f"{self.vendor_name}/{self.name}"


MODELS = [
    DeepInfraModel(
        name="Llama-3.2-11B-Vision-Instruct",
        cost=Cost(prompt=0.055e-6, completion=0.055e-6),
        context_length=128000,
        vendor_name="meta-llama",
        supports_n=False,
        properties=BASE_PROPERTIES.model_copy(update={"gdpr_compliant": False}),
    ),
    DeepInfraModel(
        name="Llama-3.2-90B-Vision-Instruct",
        cost=Cost(prompt=0.35e-6, completion=0.40e-6),
        context_length=128000,
        vendor_name="meta-llama",
        supports_n=False,
        properties=BASE_PROPERTIES.model_copy(update={"gdpr_compliant": False}),
    ),
    DeepInfraModel(
        name="Meta-Llama-3.1-405B-Instruct",
        cost=Cost(prompt=1.79e-6, completion=1.79e-6),
        context_length=32000,
        vendor_name="meta-llama",
        supports_n=False,
        properties=BASE_PROPERTIES.model_copy(update={"gdpr_compliant": False}),
    ),
    DeepInfraModel(
        name="Meta-Llama-3.1-8B-Instruct",
        cost=Cost(prompt=0.06e-6, completion=0.06e-6),
        context_length=128000,
        vendor_name="meta-llama",
        supports_n=False,
        properties=BASE_PROPERTIES.model_copy(update={"gdpr_compliant": False}),
    ),
    DeepInfraModel(
        name="Meta-Llama-3.1-70B-Instruct",
        cost=Cost(prompt=0.35e-6, completion=0.4e-6),
        context_length=128000,
        vendor_name="meta-llama",
        supports_n=False,
        properties=BASE_PROPERTIES.model_copy(
            update={"is_nsfw": True, "gdpr_compliant": False}
        ),
    ),
    DeepInfraModel(
        name="gemma-2-27b-it",
        cost=Cost(prompt=2.7e-6, completion=2.7e-6),
        context_length=4096,
        vendor_name="google",
    ),
    DeepInfraModel(
        name="gemma-2-9b-it",
        cost=Cost(prompt=0.6e-6, completion=0.6e-6),
        context_length=4096,
        vendor_name="google",
    ),
    DeepInfraModel(
        name="Mistral-7B-Instruct-v0.3",
        cost=Cost(prompt=0.055e-6, completion=0.055e-6),
        context_length=32768,
        vendor_name="mistralai",
    ),
    DeepInfraModel(
        name="Qwen2.5-72B-Instruct",
        cost=Cost(prompt=0.35e-6, completion=0.40e-6),
        context_length=32768,
        vendor_name="Qwen",
    ),
]


class DeepInfraSDKChatProviderAdapter(OpenAISDKChatAdapter):
    @staticmethod
    def get_supported_models():
        return MODELS

    @staticmethod
    def get_provider_name() -> str:
        return PROVIDER_NAME

    def get_base_sdk_url(self) -> str:
        return DEEPINFRA_BASE_URL

    @staticmethod
    def get_api_key_name() -> str:
        return API_KEY_NAME
