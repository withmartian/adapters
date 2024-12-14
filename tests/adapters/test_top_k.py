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

    adapter_response = await adapter.execute_async(
        SIMPLE_CONVERSATION_USER_ONLY, top_k=5
    )

    cassette_response = get_response_content_from_vcr(vcr, adapter)

    assert adapter_response.choices[0].message.content == cassette_response


@pytest.mark.vcr
@pytest.mark.parametrize("create_adapter", ADAPTER_COMPLETION_TEST_FACTORIES, ids=str)
async def test_async_completion(vcr: VCR, create_adapter: AdapterTestFactory) -> None:
    adapter = create_adapter()

    adapter_response = await adapter.execute_completion_async("Hi", top_k=5)

    assert adapter_response.choices[0].text
