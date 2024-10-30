from adapters.abstract_adapters.openai_sdk_chat_adapter import OpenAISDKChatAdapter
from adapters.types import Cost, Model, ModelProperties

PROVIDER_NAME = "openai"
BASE_URL = "https://api.openai.com/v1"
API_KEY_NAME = "OPENAI_API_KEY"
BASE_PROPERTIES = ModelProperties(gdpr_compliant=True)


class OpenAIModel(Model):
    vendor_name: str = PROVIDER_NAME
    provider_name: str = PROVIDER_NAME

    supports_user: bool = True
    supports_repeating_roles: bool = True
    supports_streaming: bool = True
    supports_vision: bool = True
    supports_functions: bool = True
    supports_tools: bool = True
    supports_n: bool = True
    supports_system: bool = True
    supports_multiple_system: bool = True
    supports_empty_content: bool = True
    supports_tool_choice_required: bool = True
    supports_json_output: bool = True
    supports_json_content: bool = True
    supports_last_assistant: bool = True
    supports_first_assistant: bool = True
    supports_temperature: bool = True

    properties: ModelProperties = BASE_PROPERTIES


MODELS = [
    OpenAIModel(
        name="gpt-3.5-turbo",
        cost=Cost(prompt=3.0e-6, completion=6.0e-6),
        context_length=16385,
        completion_length=16385,
        supports_vision=False,
    ),
    OpenAIModel(
        name="gpt-4",
        cost=Cost(prompt=30.0e-6, completion=60.0e-6),
        context_length=8192,
        completion_length=8192,
        supports_json_output=False,
        supports_vision=False,
    ),
    OpenAIModel(
        name="gpt-4-turbo",
        cost=Cost(prompt=10.0e-6, completion=30.0e-6),
        context_length=128000,
        completion_length=4096,
    ),
    OpenAIModel(
        name="gpt-4o",
        cost=Cost(prompt=2.5e-6, completion=10.0e-6),
        context_length=128000,
        completion_length=16384,
    ),
    OpenAIModel(
        name="gpt-4o-2024-05-13",
        cost=Cost(prompt=5.0e-6, completion=15.0e-6),
        context_length=128000,
        completion_length=4096,
    ),
    OpenAIModel(
        name="gpt-4o-2024-08-06",
        cost=Cost(prompt=2.5e-6, completion=10.0e-6),
        context_length=128000,
        completion_length=16384,
    ),
    OpenAIModel(
        name="gpt-4o-mini",
        cost=Cost(prompt=0.15e-6, completion=0.6e-6),
        context_length=128000,
        completion_length=16385,
    ),
    OpenAIModel(
        name="gpt-4o-mini-2024-07-18",
        cost=Cost(prompt=0.15e-6, completion=0.6e-6),
        context_length=128000,
        completion_length=16385,
    ),
    OpenAIModel(
        name="o1-preview",
        cost=Cost(prompt=15.0e-6, completion=60.0e-6),
        context_length=128000,
        completion_length=32768,
        supports_json_content=False,
        supports_functions=False,
        supports_tools=False,
        supports_system=False,
        supports_json_output=False,
        supports_n=False,
        supports_streaming=False,
        supports_vision=False,
        supports_temperature=False,
    ),
    OpenAIModel(
        name="o1-preview-2024-09-12",
        cost=Cost(prompt=15.0e-6, completion=60.0e-6),
        context_length=128000,
        completion_length=32768,
        supports_json_content=False,
        supports_functions=False,
        supports_tools=False,
        supports_system=False,
        supports_json_output=False,
        supports_n=False,
        supports_streaming=False,
        supports_vision=False,
        supports_temperature=False,
    ),
    OpenAIModel(
        name="o1-mini",
        cost=Cost(prompt=3.0e-6, completion=12.0e-6),
        context_length=128000,
        completion_length=65536,
        supports_json_content=False,
        supports_functions=False,
        supports_tools=False,
        supports_system=False,
        supports_json_output=False,
        supports_n=False,
        supports_streaming=False,
        supports_vision=False,
        supports_temperature=False,
    ),
    OpenAIModel(
        name="o1-mini-2024-09-12",
        cost=Cost(prompt=3.0e-6, completion=12.0e-6),
        context_length=128000,
        completion_length=65536,
        supports_json_content=False,
        supports_functions=False,
        supports_tools=False,
        supports_system=False,
        supports_json_output=False,
        supports_n=False,
        supports_streaming=False,
        supports_vision=False,
        supports_temperature=False,
    ),
]


class OpenAISDKChatProviderAdapter(OpenAISDKChatAdapter):
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
