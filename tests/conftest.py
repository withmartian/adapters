import asyncio

import pytest

import adapters.utils.network_utils as network_utils


@pytest.fixture()
async def mock_async_request(monkeypatch, request):
    return_value = request.param

    async def mock_return(*args, **kwargs):  # pylint: disable=unused-argument
        return return_value

    monkeypatch.setattr(network_utils, "async_send_request", mock_return)


@pytest.fixture()
def mock_sync_request(monkeypatch, request):
    return_value = request.param
    monkeypatch.setattr(
        network_utils,
        "send_request",
        lambda *args, **kwargs: return_value,
    )


@pytest.fixture(scope="module")
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


@pytest.fixture(scope="session")
def event_loop():
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
    yield loop
    loop.close()
