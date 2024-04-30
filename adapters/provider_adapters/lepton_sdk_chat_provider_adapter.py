"""Notes
- Context length not found in the lepton docs.
- Each model has it own base url.
"""
import re
from typing import Pattern

from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    Choice,
    ChoiceDelta,
)

from adapters.abstract_adapters.openai_sdk_chat_adapter import OpenAISDKChatAdapter
from adapters.abstract_adapters.provider_adapter_mixin import ProviderAdapterMixin
from adapters.types import Cost, Model

_PROVIDER_NAME = "lepton"
_API_KEY_NAME = "LEPTON_API_TOKEN"
_API_KEY_PATTERN = re.compile(r".*")


class _LeptonModel(Model):
    base_url: str
    provider_name: str = _PROVIDER_NAME
    supports_streaming: bool = True


_MODELS: dict[str, Model] = {
    "gemma-7b": _LeptonModel(
        base_url="https://gemma-7b.lepton.run/api/v1/",
        name="gemma-7b",
        cost=Cost(prompt=0.10e-6, completion=0.10e-6),
        context_length=8192,
        vendor_name="",
    ),
    "mistral-7b": _LeptonModel(
        base_url="https://mistral-7b.lepton.run/api/v1/",
        name="mistral-7b",
        cost=Cost(prompt=0.11e-6, completion=0.11e-6),
        context_length=8192,  # https://arxiv.org/pdf/2310.06825.pdf
        vendor_name="",
    ),
    "mixtral-8x22b": _LeptonModel(
        base_url="https://mixtral-8x22b.lepton.run/api/v1/",
        name="mixtral-8x22b",
        cost=Cost(prompt=0.50e-6, completion=0.50e-6),
        context_length=65536,  # https://huggingface.co/v2ray/Mixtral-8x22B-v0.1/discussions/5
        vendor_name="",
    ),
    "mixtral-8x7b": _LeptonModel(
        base_url="https://mixtral-8x7b.lepton.run/api/v1/",
        name="mixtral-8x7b",
        cost=Cost(prompt=0.80e-6, completion=0.80e-6),
        context_length=32768,  # https://mistral.ai/news/mixtral-of-experts/
        vendor_name="",
    ),
}


class LeptonSDKChatProviderAdapter(ProviderAdapterMixin, OpenAISDKChatAdapter):
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.base_url = self.get_model_base_url(model_name)
        super().__init__()

    @staticmethod
    def get_supported_models() -> list[Model]:
        return list(_MODELS.values())

    @staticmethod
    def get_provider_name() -> str:
        return _PROVIDER_NAME

    @staticmethod
    def get_api_key_name() -> str:
        return _API_KEY_NAME

    @staticmethod
    def get_api_key_pattern() -> Pattern:
        return _API_KEY_PATTERN

    def get_base_sdk_url(self) -> str:  # type: ignore[override]  # pylint: disable=arguments-differ
        return self.base_url

    def get_model_base_url(self, model_name: str) -> str:
        if model_name in _MODELS:
            return _MODELS[model_name].base_url  # type: ignore[attr-defined]
        raise ValueError(f"Model with name {model_name} not found in supported models.")

    def extract_stream_response(self, request, response: ChatCompletionChunk) -> str:
        if (
            not response.choices
        ):  # It must be the last response from Lepton that is empty.
            response.choices = [
                Choice(
                    delta=ChoiceDelta(),
                    finish_reason="stop",
                    index=0,
                ),
            ]
        return f"data: {response.model_dump_json()}\n\n"
