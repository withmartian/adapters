import pytest

from tests.utils import (
    ADAPTER_TEST_FACTORIES,
    SIMPLE_CONVERSATION_USER_ONLY,
    AdapterTestFactory,
)
from vcr import VCR


@pytest.mark.vcr
@pytest.mark.parametrize("create_adapter", ADAPTER_TEST_FACTORIES, ids=str)
def test_sync(vcr: VCR, create_adapter: AdapterTestFactory) -> None:
    adapter = create_adapter()

    if not adapter.get_model().supports_streaming:
        return

    adapter_response = adapter.execute_sync(
        SIMPLE_CONVERSATION_USER_ONLY,
        stream=True,
    )

    chunks = list(adapter_response.response)

    response = "".join(
        [
            chunk.choices[0].delta.content
            for chunk in chunks
            if chunk.choices[0].delta.content
        ]
    )
    assert len(response) > 0


@pytest.mark.vcr
@pytest.mark.parametrize("create_adapter", ADAPTER_TEST_FACTORIES, ids=str)
async def test_async(vcr: VCR, create_adapter: AdapterTestFactory) -> None:
    adapter = create_adapter()

    if not adapter.get_model().supports_streaming:
        return

    adapter_response = await adapter.execute_async(
        SIMPLE_CONVERSATION_USER_ONLY,
        stream=True,
    )

    chunks = [data_chunk async for data_chunk in adapter_response.response]

    response = "".join(
        [
            chunk.choices[0].delta.content
            for chunk in chunks
            if chunk.choices[0].delta.content
        ]
    )
    assert len(response) > 0
