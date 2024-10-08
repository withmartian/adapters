from abc import abstractmethod
import os
from typing import List


class ApiKeyAdapterMixin:
    _api_key: str = ""
    _api_keys: List[str] = []
    _next_api_key = 0

    def __init__(self):
        api_keys = os.environ.get(f"{self.get_api_key_name()}_LIST", None)
        if api_keys:
            ApiKeyAdapterMixin._api_keys = api_keys.split(",")
        else:
            self._api_keys = [os.environ.get(self.get_api_key_name())]

    @staticmethod
    @abstractmethod
    def get_api_key_name() -> str:
        """returns the api key name for the adapter

        Returns:
            str: api key name for the adapter (e.g. "OPENAI_API_KEY")
        """

    def _api_key_was_set_by_user(self):
        return self._api_key

    def set_api_key(self, api_key: str) -> None:
        """sets an api key for the adapter

        Args:
            api_key: api key for the adapter
        """
        if not api_key:
            raise ValueError("api_key cannot be empty")
        self._api_key = api_key

    def get_api_key(self) -> str:
        """gets an api key for the adapter

        Returns:
            str: api key for the adapter if it exists, else the env default api key
        """

        if not self._api_key_was_set_by_user() and self._api_keys:
            api_key = self._api_keys[self._next_api_key]
            self._next_api_key += 1
            if len(self._api_keys) == self._next_api_key:
                self._next_api_key = 0
            return api_key
        return self._api_key
