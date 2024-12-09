from adapters.abstract_adapters.openai_sdk_chat_adapter import OpenAISDKChatAdapter
from adapters.types import Cost, Model, Provider, Vendor


class LeptonModel(Model):
    provider_name: str = Provider.lepton.value

    supports_vision: bool = False
    supports_json_content: bool = False

    can_assistant_first: bool = False
    can_assistant_last: bool = False
    can_assistant_only: bool = False

    can_system_multiple: bool = False
    can_repeating_roles: bool = False


MODELS: list[Model] = [
    LeptonModel(
        name="mistral-7b",
        cost=Cost(prompt=0.07e-6, completion=0.07e-6),
        context_length=8192,
        vendor_name=Vendor.mistralai.value,
    ),
    LeptonModel(
        name="mixtral-8x7b",
        cost=Cost(prompt=0.50e-6, completion=0.50e-6),
        context_length=32768,
        vendor_name=Vendor.mistralai.value,
    ),
    LeptonModel(
        name="qwen2-72b",
        cost=Cost(prompt=0.8e-6, completion=0.8e-6),
        context_length=128000,
        vendor_name=Vendor.qwen.value,
    ),
    LeptonModel(
        name="wizardlm-2-7b",
        cost=Cost(prompt=0.07e-6, completion=0.07e-6),
        context_length=32000,
        vendor_name=Vendor.wizardlm.value,
    ),
    LeptonModel(
        name="wizardlm-2-8x22b",
        cost=Cost(prompt=1.0e-6, completion=1.0e-6),
        context_length=64000,
        vendor_name=Vendor.wizardlm.value,
        supports_tools_choice_required=False,
    ),
    LeptonModel(
        name="dolphin-mixtral-8x7b",
        cost=Cost(prompt=0.5e-6, completion=0.5e-6),
        context_length=32000,
        vendor_name=Vendor.mistralai.value,
    ),
]


class LeptonSDKChatProviderAdapter(OpenAISDKChatAdapter):
    @staticmethod
    def get_supported_models() -> list[Model]:
        return MODELS

    @staticmethod
    def get_api_key_name() -> str:
        return "LEPTON_API_KEY"

    def get_base_sdk_url(self) -> str:
        return f"https://{self.get_model().name if self._current_model else ''}.lepton.run/api/v1/"

    def _set_current_model(self, model: Model) -> None:
        super()._set_current_model(model)

        self._setup_clients(self.get_api_key())
