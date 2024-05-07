from abc import abstractmethod
from typing import List, Optional

from adapters.types import Model


class ProviderAdapterMixin:
    _current_model: Optional[Model] = None

    def _set_current_model(self, model: Model) -> None:
        self._current_model = model

    @staticmethod
    @abstractmethod
    def get_supported_models() -> List[Model]:
        pass

    def get_model(self) -> Model:
        if self._current_model is None:
            raise ValueError("Model is not set")
        return self._current_model
