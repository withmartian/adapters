"""Notes
- Context length not found in the lepton docs.
- Each model has it own base url.
"""

from httpx import URL

from adapters.abstract_adapters.openai_sdk_chat_adapter import OpenAISDKChatAdapter
from adapters.types import Cost, Model, ModelProperties

PROVIDER_NAME = "lepton"
BASE_URL = "https://{}.lepton.run/api/v1/"
API_KEY_NAME = "LEPTON_API_KEY"
BASE_PROPERTIES = ModelProperties(
    open_source=True,
    gdpr_compliant=True,
)


class LeptonModel(Model):
    base_url: str
    provider_name: str = PROVIDER_NAME

    supports_repeating_roles: bool = True
    supports_system: bool = True
    supports_multiple_system: bool = True
    supports_empty_content: bool = True
    supports_tool_choice_required: bool = True
    supports_last_assistant: bool = True
    supports_first_assistant: bool = True
    supports_streaming: bool = True
    supports_temperature: bool = True

    properties: ModelProperties = BASE_PROPERTIES


# TODO: add more models
MODELS = [
    LeptonModel(
        base_url=BASE_URL.format("mistral-7b"),
        name="mistral-7b",
        cost=Cost(prompt=0.07e-6, completion=0.07e-6),
        context_length=8192,
        vendor_name="mistralai",
    ),
    LeptonModel(
        base_url=BASE_URL.format("mixtral-8x7b"),
        name="mixtral-8x7b",
        cost=Cost(prompt=0.50e-6, completion=0.50e-6),
        context_length=32768,
        vendor_name="mistralai",
    ),
    LeptonModel(
        base_url=BASE_URL.format("qwen2-72b"),
        name="qwen2-72b",
        cost=Cost(prompt=0.8e-6, completion=0.8e-6),
        context_length=128000,
        vendor_name="qwen",
    ),
    LeptonModel(
        base_url=BASE_URL.format("wizardlm-2-7b"),
        name="wizardlm-2-7b",
        cost=Cost(prompt=0.07e-6, completion=0.07e-6),
        context_length=32000,
        vendor_name="wizardlm",
    ),
    LeptonModel(
        base_url=BASE_URL.format("wizardlm-2-8x22b"),
        name="wizardlm-2-8x22b",
        cost=Cost(prompt=1.0e-6, completion=1.0e-6),
        context_length=64000,
        vendor_name="wizardlm",
    ),
    LeptonModel(
        base_url=BASE_URL.format("dolphin-mixtral-8x7b"),
        name="dolphin-mixtral-8x7b",
        cost=Cost(prompt=0.5e-6, completion=0.5e-6),
        context_length=32000,
        vendor_name="mistralai",
    ),
]


class LeptonSDKChatProviderAdapter(OpenAISDKChatAdapter):
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

    def get_base_sdk_url(self) -> str:
        return BASE_URL

    def _set_current_model(self, model: Model) -> None:
        super()._set_current_model(model)

        self._client_sync.base_url = URL(self._current_model.base_url)
        self._client_async.base_url = URL(self._current_model.base_url)

    # def extract_stream_response(self, request, response: ChatCompletionChunk) -> str:
    #     # It must be the last response from Lepton that is empty.
    #     if not response.choices:
    #         response.choices = [
    #             Choice(
    #                 delta=ChoiceDelta(),
    #                 finish_reason="stop",
    #                 index=0,
    #             ),
    #         ]
    #     elif response.choices[0].delta.content is None:
    #         # It must be the first response.
    #         # Most models start with an empty string.
    #         response.choices[0].delta.content = ""

    #     return f"data: {response.model_dump_json()}\n\n"
