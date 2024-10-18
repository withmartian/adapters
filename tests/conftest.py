import pytest


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
