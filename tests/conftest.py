import asyncio

import pytest
from aiolimiter import AsyncLimiter

import adapters.utils.network_utils as network_utils
from adapters.rate_limiter import OpenAIModelRateLimiter
from tests.utils import ASYNC_LIMITER_LEAK_BUCKET_TIME, setup_tiktoken_cache


@pytest.fixture(scope="session", autouse=True)
def do_tiktoken_cache():
    setup_tiktoken_cache()


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


@pytest.fixture()
def openai_1rpm_rate_limiter(monkeypatch, request):
    class_to_mock = request.param

    def mock_init(self, rpm: int = 0):
        self.limiter = (
            AsyncLimiter(max_rate=1, time_period=ASYNC_LIMITER_LEAK_BUCKET_TIME)
            if rpm > 0
            else None
        )

    monkeypatch.setattr(OpenAIModelRateLimiter, "__init__", mock_init)
    limiter = OpenAIModelRateLimiter(rpm=1)
    monkeypatch.setattr(
        class_to_mock,
        "get_rate_limiter",
        lambda *args, **kwargs: limiter,
    )


@pytest.fixture(scope="module")
def vcr_config():
    return {
        "filter_headers": ["authorization", "x-api-key", "X-API-Key"],
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
