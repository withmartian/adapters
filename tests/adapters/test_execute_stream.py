import pytest

from adapters.abstract_adapters.base_adapter import BaseAdapter
from tests.utils import (
    TEST_ADAPTERS,
    SIMPLE_CONVERSATION_USER_ONLY,
)
from vcr import VCR


@pytest.mark.vcr
@pytest.mark.parametrize("adapter", TEST_ADAPTERS, ids=lambda adapter: str(adapter))
def test_sync(vcr: VCR, adapter: BaseAdapter) -> None:
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
@pytest.mark.parametrize("adapter", TEST_ADAPTERS, ids=lambda adapter: str(adapter))
async def test_async(vcr: VCR, adapter: BaseAdapter) -> None:
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
