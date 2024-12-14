import pytest

from tests.utils import (
    ADAPTER_CHAT_TEST_FACTORIES,
    ADAPTER_COMPLETION_TEST_FACTORIES,
    SIMPLE_CONVERSATION_USER_ONLY,
    AdapterTestFactory,
    get_response_content_from_vcr,
)
from vcr import VCR


@pytest.mark.vcr
@pytest.mark.parametrize("create_adapter", ADAPTER_CHAT_TEST_FACTORIES, ids=str)
async def test_async(vcr: VCR, create_adapter: AdapterTestFactory) -> None:
    adapter = create_adapter()

    if not adapter.get_model().supports_stop:
        return

    adapter_response = await adapter.execute_async(
        SIMPLE_CONVERSATION_USER_ONLY, stop="stop"
    )

    cassette_response = get_response_content_from_vcr(vcr, adapter)

    assert adapter_response.choices[0].message.content == cassette_response


@pytest.mark.vcr
@pytest.mark.parametrize("create_adapter", ADAPTER_CHAT_TEST_FACTORIES, ids=str)
async def test_async_list(vcr: VCR, create_adapter: AdapterTestFactory) -> None:
    adapter = create_adapter()

    if not adapter.get_model().supports_stop:
        return

    adapter_response = await adapter.execute_async(
        SIMPLE_CONVERSATION_USER_ONLY, stop=["stop", "end"]
    )

    cassette_response = get_response_content_from_vcr(vcr, adapter)

    assert adapter_response.choices[0].message.content == cassette_response


@pytest.mark.vcr
@pytest.mark.parametrize("create_adapter", ADAPTER_COMPLETION_TEST_FACTORIES, ids=str)
async def test_async_completion(vcr: VCR, create_adapter: AdapterTestFactory) -> None:
    adapter = create_adapter()

    if not adapter.get_model().supports_stop:
        return

    adapter_response = await adapter.execute_completion_async("Hi", stop="stop")

    assert adapter_response.choices[0].text


@pytest.mark.vcr
@pytest.mark.parametrize("create_adapter", ADAPTER_COMPLETION_TEST_FACTORIES, ids=str)
async def test_async_list_completion(
    vcr: VCR, create_adapter: AdapterTestFactory
) -> None:
    adapter = create_adapter()

    if not adapter.get_model().supports_stop:
        return

    adapter_response = await adapter.execute_completion_async(
        "Hi", stop=["stop", "end"]
    )

    assert adapter_response.choices[0].text
