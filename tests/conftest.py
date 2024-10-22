import pytest


@pytest.fixture(name="adapters_patch", autouse=True)
def fixture_adapters_patch(monkeypatch):
    monkeypatch.setattr(
        "adapters.client_cache.client_cache.get_client",
        lambda base_url, api_key, mode: None,
    )


@pytest.fixture(scope="session")
def vcr_config():
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
