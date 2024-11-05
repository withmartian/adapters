import pytest

from adapters.abstract_adapters.base_adapter import BaseAdapter
from tests.utils import (
    TEST_ADAPTERS,
    SIMPLE_CONVERSATION_USER_ONLY,
    get_response_content_from_vcr,
)
from vcr import VCR


@pytest.mark.vcr
@pytest.mark.parametrize("adapter", TEST_ADAPTERS, ids=lambda adapter: str(adapter))
async def test_async(vcr: VCR, adapter: BaseAdapter) -> None:
    if not adapter.get_model().supports_n:
        return

    n = 2

    adapter_response = await adapter.execute_async(SIMPLE_CONVERSATION_USER_ONLY, n=n)

    cassette_response = get_response_content_from_vcr(vcr, adapter)

    assert adapter_response.choices[0].message.content == cassette_response
    assert len(adapter_response.choices) == n
