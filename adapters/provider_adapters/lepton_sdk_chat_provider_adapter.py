"""Notes
- Context length not found in the lepton docs.
- Each model has it own base url.
"""

import re
from typing import Pattern

from httpx import URL
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    Choice,
    ChoiceDelta,
)

from adapters.abstract_adapters.openai_sdk_chat_adapter import OpenAISDKChatAdapter
from adapters.abstract_adapters.provider_adapter_mixin import ProviderAdapterMixin
from adapters.types import Cost, Model

PROVIDER_NAME = "lepton"
BASE_URL = "https://{}.lepton.run/api/v1/"
API_KEY_NAME = "LEPTON_API_KEY"
API_KEY_PATTERN = re.compile(r".*")


class LeptonModel(Model):
    base_url: str
    provider_name: str = PROVIDER_NAME
    supports_streaming: bool = True


MODELS = [
    LeptonModel(
        base_url=BASE_URL.format("gemma-7b"),
        name="gemma-7b",
        cost=Cost(prompt=0.10e-6, completion=0.10e-6),
        context_length=8192,
        vendor_name="google",
    ),
    LeptonModel(
        base_url=BASE_URL.format("mistral-7b"),
        name="mistral-7b",
        cost=Cost(prompt=0.11e-6, completion=0.11e-6),
        context_length=8192,
        vendor_name="mistralai",
    ),
    LeptonModel(
        base_url=BASE_URL.format("mixtral-8x22b"),
        name="mixtral-8x22b",
        cost=Cost(prompt=0.80e-6, completion=0.80e-6),
        context_length=65536,
        vendor_name="mistralai",
    ),
    LeptonModel(
        base_url=BASE_URL.format("mixtral-8x7b"),
        name="mixtral-8x7b",
        cost=Cost(prompt=0.50e-6, completion=0.50e-6),
        context_length=32768,
        vendor_name="mistralai",
    ),
]


class LeptonSDKChatProviderAdapter(ProviderAdapterMixin, OpenAISDKChatAdapter):
    _current_model: LeptonModel

    @staticmethod
    def get_supported_models():
        return MODELS

    @staticmethod
    def get_provider_name() -> str:
        return PROVIDER_NAME

    @staticmethod
    def get_api_key_name() -> str:
        return API_KEY_NAME

    @staticmethod
    def get_api_key_pattern() -> Pattern:
        return API_KEY_PATTERN

    @staticmethod
    def get_base_sdk_url() -> str:
        return BASE_URL

    def _init_current_model(self, model_name: str) -> None:
        super()._init_current_model(model_name)

        self._sync_client.base_url = URL(self._current_model.base_url)
        self._async_client.base_url = URL(self._current_model.base_url)

    def get_model_name(self) -> str:
        if self._current_model is None:
            raise ValueError("Model not set")
        return self._current_model.name

    # It must be the last response from Lepton that is empty.
    def extract_stream_response(self, request, response: ChatCompletionChunk) -> str:
        if not response.choices:
            response.choices = [
                Choice(
                    delta=ChoiceDelta(),
                    finish_reason="stop",
                    index=0,
                ),
            ]
        return f"data: {response.model_dump_json()}\n\n"
