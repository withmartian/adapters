import re
from typing import Pattern

from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

from adapters.abstract_adapters.openai_sdk_chat_adapter import OpenAISDKChatAdapter
from adapters.abstract_adapters.provider_adapter_mixin import ProviderAdapterMixin
from adapters.types import Cost, Model, ModelPredicates

PROVIDER_NAME = "fireworks"
BASE_URL = "https://api.fireworks.ai/inference/v1"
API_KEY_NAME = "FIREWORKS_API_KEY"
API_KEY_PATTERN = re.compile(r".*")
BASE_PREDICATES = ModelPredicates(
    open_source=True,
    gdpr_compliant=True,
)


class FireworksModel(Model):
    supports_streaming: bool = True
    provider_name: str = PROVIDER_NAME
    predicates: ModelPredicates = BASE_PREDICATES

    def _get_api_path(self) -> str:
        return f"{self.vendor_name}/{self.name}"


MODELS = [
    FireworksModel(
        name="gemma2-9b-it",
        cost=Cost(prompt=0.2e-6, completion=0.2e-6),
        context_length=8192,
        vendor_name="accounts/fireworks/models",
        supports_system=False,
        supports_first_assistant=False,
    ),
    FireworksModel(
        name="llama-v3-8b-instruct",
        cost=Cost(prompt=0.2e-6, completion=0.2e-6),
        context_length=8192,
        vendor_name="accounts/fireworks/models",
        predicates=BASE_PREDICATES.model_copy(update={"gdpr_compliant": False}),
    ),
    FireworksModel(
        name="llama-v3-70b-instruct",
        cost=Cost(prompt=0.9e-6, completion=0.9e-6),
        context_length=8192,
        vendor_name="accounts/fireworks/models",
        predicates=BASE_PREDICATES.model_copy(
            update={"is_nsfw": True, "gdpr_compliant": False}
        ),
    ),
    FireworksModel(
        name="mixtral-8x22b-instruct",
        cost=Cost(prompt=0.9e-6, completion=0.9e-6),
        context_length=65_536,
        vendor_name="accounts/fireworks/models",
    ),
    FireworksModel(
        name="mixtral-8x7b-instruct",
        cost=Cost(prompt=0.5e-6, completion=0.5e-6),
        context_length=32_768,
        vendor_name="accounts/fireworks/models",
        supports_first_assistant=False,
    ),
    FireworksModel(
        name="llama-v3p1-405b-instruct",
        cost=Cost(prompt=3.0e-6, completion=3.0e-6),
        context_length=131072,
        vendor_name="accounts/fireworks/models",
        predicates=BASE_PREDICATES.model_copy(update={"gdpr_compliant": False}),
    ),
    FireworksModel(
        name="llama-v3p1-70b-instruct",
        cost=Cost(prompt=0.9e-6, completion=0.9e-6),
        context_length=131072,
        vendor_name="accounts/fireworks/models",
        predicates=BASE_PREDICATES.model_copy(
            update={"is_nsfw": True, "gdpr_compliant": False}
        ),
    ),
    FireworksModel(
        name="llama-v3p1-8b-instruct",
        cost=Cost(prompt=0.2e-6, completion=0.2e-6),
        context_length=131072,
        vendor_name="accounts/fireworks/models",
        predicates=BASE_PREDICATES.model_copy(update={"gdpr_compliant": False}),
    ),
]


class FireworksSDKChatProviderAdapter(ProviderAdapterMixin, OpenAISDKChatAdapter):
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

    @staticmethod
    def get_api_key_pattern() -> Pattern:
        return API_KEY_PATTERN

    def extract_stream_response(self, request, response: ChatCompletionChunk) -> str:
        if response.choices and response.choices[0].delta.content is None:
            # It must be the first response.
            # Most models start with an empty string.
            response.choices[0].delta.content = ""

        return f"data: {response.model_dump_json()}\n\n"
