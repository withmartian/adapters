from typing import Any
import pytest


@pytest.fixture(name="adapters_patch", autouse=True, scope="function")
def fixture_adapters_patch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "adapters.client_cache.client_cache.get_client",
        lambda base_url, api_key, mode: None,
    )


@pytest.fixture(scope="session")
def vcr_config() -> dict[str, Any]:
    return {
        "filter_headers": [
            "authorization",
            "x-api-key",
            "X-API-Key",
            "x-goog-api-key",
            "Set-Cookie",
            "cookie",
        ],
        "decode_compressed_response": True,
    }
