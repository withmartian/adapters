import pytest

from tests.utils import (
    ADAPTER_TEST_FACTORIES,
    SIMPLE_CONVERSATION_USER_ONLY,
    AdapterTestFactory,
    get_response_content_from_vcr,
)
from vcr import VCR


@pytest.mark.vcr
@pytest.mark.parametrize("create_adapter", ADAPTER_TEST_FACTORIES, ids=str)
async def test_async(vcr: VCR, create_adapter: AdapterTestFactory) -> None:
    adapter = create_adapter()

    if not adapter.get_model().supports_n:
        return

    n = 2

    adapter_response = await adapter.execute_async(SIMPLE_CONVERSATION_USER_ONLY, n=n)

    cassette_response = get_response_content_from_vcr(vcr, adapter)

    assert adapter_response.choices[0].message.content == cassette_response
    assert len(adapter_response.choices) == n
