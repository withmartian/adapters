from adapters.abstract_adapters.openai_sdk_chat_adapter import OpenAISDKChatAdapter
from adapters.types import Cost, Model, ModelProperties

PROVIDER_NAME = "fireworks"
BASE_URL = "https://api.fireworks.ai/inference/v1"
API_KEY_NAME = "FIREWORKS_API_KEY"
BASE_PROPERTIES = ModelProperties(
    open_source=True,
)


class FireworksModel(Model):
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
        return f"accounts/fireworks/models/{self.name}"


MODELS = [
    FireworksModel(
        name="llama-v3p1-405b-instruct",
        cost=Cost(prompt=3.00e-6, completion=3.00e-6),
        context_length=131072,
        vendor_name="meta-llama",
    ),
    FireworksModel(
        name="llama-v3p1-70b-instruct",
        cost=Cost(prompt=0.90e-6, completion=0.90e-6),
        context_length=131072,
        vendor_name="meta-llama",
    ),
    FireworksModel(
        name="llama-v3p1-8b-instruct",
        cost=Cost(prompt=0.20e-6, completion=0.20e-6),
        context_length=131072,
        vendor_name="meta-llama",
    ),
    FireworksModel(
        name="llama-v3p2-3b-instruct",
        cost=Cost(prompt=0.10e-6, completion=0.10e-6),
        context_length=131072,
        vendor_name="meta-llama",
    ),
    FireworksModel(
        name="llama-v3p2-11b-vision-instruct",
        cost=Cost(prompt=0.20e-6, completion=0.20e-6),
        context_length=131072,
        vendor_name="meta-llama",
    ),
    FireworksModel(
        name="llama-v3p2-1b-instruct",
        cost=Cost(prompt=0.10e-6, completion=0.10e-6),
        context_length=131072,
        vendor_name="meta-llama",
    ),
    FireworksModel(
        name="llama-v3p2-90b-vision-instruct",
        cost=Cost(prompt=0.90e-6, completion=0.90e-6),
        context_length=131072,
        vendor_name="meta-llama",
    ),
    FireworksModel(
        name="qwen2p5-72b-instruct",
        cost=Cost(prompt=0.90e-6, completion=0.90e-6),
        context_length=32768,
        vendor_name="qwen",
    ),
    FireworksModel(
        name="mixtral-8x22b-instruct",
        cost=Cost(prompt=1.20e-6, completion=1.20e-6),
        context_length=65536,
        vendor_name="mistralai",
    ),
    FireworksModel(
        name="mixtral-8x7b-instruct",
        cost=Cost(prompt=0.50e-6, completion=0.50e-6),
        context_length=32768,
        vendor_name="mistralai",
    ),
]


class FireworksSDKChatProviderAdapter(OpenAISDKChatAdapter):
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

    # def extract_stream_response(self, request, response: ChatCompletionChunk) -> str:
    #     if response.choices and response.choices[0].delta.content is None:
    #         # It must be the first response.
    #         # Most models start with an empty string.
    #         response.choices[0].delta.content = ""

    #     return f"data: {response.model_dump_json()}\n\n"
