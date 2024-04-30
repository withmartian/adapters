from abc import abstractmethod
from typing import List, Optional

from adapters.types import Cost, Model


class ProviderAdapterMixin:
    _current_model: Optional[Model] = None

    """_summary_
        init the current model as this is a meta adapter.
        It will lock the current model to the one specified model
        That cannot be changed later on for consistency.
        Args:
            model_name (_type_): _description_

        Raises:
            ValueError: _description_
        """

    def _init_current_model(self, model_name) -> None:
        if self._current_model is not None:
            raise ValueError("Current model already set")

        for model in self.get_supported_models():
            if model_name in (
                f"{self.get_provider_name()}/{model.vendor_name}/{model.name}",
                f"{model.vendor_name}/{model.name}",
                model.name,
            ):
                self._current_model = model
                return
        raise ValueError(f"Model {model_name} not supported")

    def get_current_model(self) -> Optional[Model]:
        return self._current_model

    @staticmethod
    @abstractmethod
    def get_supported_models() -> List[Model]:
        pass

    def get_context_length(self) -> int:
        if self._current_model is None:
            raise ValueError("Model not set")
        return self._current_model.context_length

    def get_completion_length(self) -> Optional[int]:
        if self._current_model is None:
            raise ValueError("Model not set")
        return self._current_model.completion_length

    def get_token_cost(self) -> Cost:
        if self._current_model is None:
            raise ValueError("Model not set")
        return self._current_model.cost.copy(deep=True)

    def get_name(self) -> str:
        if self._current_model is None:
            raise ValueError("Model not set")
        return f"{self.get_provider_name()}/{self._current_model.vendor_name}/{self._current_model.name}"

    def get_model_name(self) -> str:
        if self._current_model is None:
            raise ValueError("Model not set")
        return f"{self._current_model.vendor_name}/{self._current_model.name}"

    @staticmethod
    @abstractmethod
    def get_provider_name() -> str:
        pass

    def supports_streaming(self) -> bool:
        if self._current_model is None:
            return False
        return self._current_model.supports_streaming

    def supports_vision(self) -> bool:
        if self._current_model is None:
            return False
        return self._current_model.supports_vision

    def supports_functions(self) -> bool:
        if self._current_model is None:
            return False
        return self._current_model.supports_functions

    def supports_tools(self) -> bool:
        if self._current_model is None:
            return False
        return self._current_model.supports_tools

    def supports_n(self) -> bool:
        if self._current_model is None:
            return False
        return self._current_model.supports_n

    def supports_json_output(self) -> bool:
        if self._current_model is None:
            return False
        return self._current_model.supports_json_output

    def supports_json_content(self) -> bool:
        if self._current_model is None:
            return False
        return self._current_model.supports_json_content
