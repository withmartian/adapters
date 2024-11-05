import pytest

from adapters.abstract_adapters.base_adapter import BaseAdapter
from tests.utils import (
    TEST_ADAPTERS,
    SIMPLE_CONVERSATION_JSON_CONTENT,
    get_response_content_from_vcr,
)
from vcr import VCR


@pytest.mark.parametrize("adapter", TEST_ADAPTERS)
@pytest.mark.vcr
async def test_async(vcr: VCR, adapter: BaseAdapter) -> None:
    adapter_response = await adapter.execute_async(SIMPLE_CONVERSATION_JSON_CONTENT)

    cassette_response = get_response_content_from_vcr(vcr, adapter)

    assert adapter_response.choices[0].message.content == cassette_response
