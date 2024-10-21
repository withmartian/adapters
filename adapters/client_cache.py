from typing import Any, Dict, Literal


# TODO: Add time based expiration if leaks connections or memory
class ClientCache:
    def __init__(self):
        self._client_cache: Dict[str, Any] = {}

    def get_client(
        self, base_url: str, api_key: str, mode: Literal["sync", "async"]
    ) -> Any:
        return self._client_cache.get(f"{base_url}-{api_key}-{mode}")

    def set_client(
        self, base_url: str, api_key: str, mode: Literal["sync", "async"], client: Any
    ) -> None:
        self._client_cache[f"{base_url}-{api_key}-{mode}"] = client


client_cache = ClientCache()
