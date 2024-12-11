import pytest

from tests.utils import (
    ADAPTER_COMPLETION_TEST_FACTORIES,
    AdapterTestFactory,
)
from vcr import VCR


@pytest.mark.vcr
@pytest.mark.parametrize("create_adapter", ADAPTER_COMPLETION_TEST_FACTORIES, ids=str)
async def test_async(vcr: VCR, create_adapter: AdapterTestFactory) -> None:
    adapter = create_adapter()

    if not (
        adapter.get_model().supports_completion
        and adapter.get_model().supports_streaming
    ):
        return

    adapter_response = await adapter.execute_completion_async(
        prompt="Hi", max_tokens=10, stream=True
    )

    chunks = [data_chunk async for data_chunk in adapter_response.response]

    response = "".join(
        [chunk.choices[0].text for chunk in chunks if len(chunk.choices)]
    )
    assert len(response)


@pytest.mark.vcr
@pytest.mark.parametrize("create_adapter", ADAPTER_COMPLETION_TEST_FACTORIES, ids=str)
def test_sync(vcr: VCR, create_adapter: AdapterTestFactory) -> None:
    adapter = create_adapter()

    if not (
        adapter.get_model().supports_completion
        and adapter.get_model().supports_streaming
    ):
        return

    adapter_response = adapter.execute_completion_sync(
        prompt="Hi", max_tokens=10, stream=True
    )

    chunks = list(adapter_response.response)

    response = "".join(
        [chunk.choices[0].text for chunk in chunks if len(chunk.choices)]
    )
    assert len(response)
