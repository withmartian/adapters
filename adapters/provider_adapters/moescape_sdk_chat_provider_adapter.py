from adapters.abstract_adapters.openai_sdk_chat_adapter import OpenAISDKChatAdapter
from adapters.types import Cost, Model, Provider, Vendor


class MoescapeModel(Model):
    provider_name: str = Provider.lepton.value
    vendor_name: str = Vendor.moescape.value


MODELS: list[Model] = [
    MoescapeModel(
        name="nephara",
        api_name="llama3-8b-instruct",
        cost=Cost(prompt=0, completion=0),
        context_length=8192,
    ),
    MoescapeModel(
        name="lunara",
        api_name="lunara-12bv1",
        cost=Cost(prompt=0, completion=0),
        context_length=12288,
        can_system_only=False,
    ),
    MoescapeModel(
        name="seth",
        api_name="llama3-70b-2",
        cost=Cost(prompt=0, completion=0),
        context_length=32768,
        can_system_only=False,
    ),
]


class MoescapeSDKChatProviderAdapter(OpenAISDKChatAdapter):
    @staticmethod
    def get_supported_models() -> list[Model]:
        return MODELS

    @staticmethod
    def get_api_key_name() -> str:
        return "MOESCAPE_API_KEY"

    def get_base_sdk_url(self) -> str:
        return f"https://jtwtcue2-{self._current_model.api_name if self._current_model and self._current_model.api_name else ''}.tin.lepton.run/api/v1/"

    def _set_current_model(self, model: Model) -> None:
        super()._set_current_model(model)

        self._setup_clients(self.get_api_key())
