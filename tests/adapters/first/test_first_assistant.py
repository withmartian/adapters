import pytest

from adapters.abstract_adapters.base_adapter import BaseAdapter
from tests.utils import (
    ADAPTERS,
    SIMPLE_CONVERSATION_USER_ONLY,
    TEST_TEMPERATURE,
    get_response_content_from_vcr,
)


@pytest.mark.parametrize("adapter", ADAPTERS)
@pytest.mark.vcr
async def test_async(vcr, adapter: BaseAdapter):
    adapter_response = await adapter.execute_async(
        SIMPLE_CONVERSATION_USER_ONLY, temperature=TEST_TEMPERATURE
    )

    cassette_response = get_response_content_from_vcr(vcr, adapter)

    assert adapter_response.choices[0].message.content == cassette_response
