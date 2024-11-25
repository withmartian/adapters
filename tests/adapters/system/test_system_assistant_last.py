import pytest

from tests.utils import (
    ADAPTER_TEST_FACTORIES,
    AdapterTestFactory,
    get_response_content_from_vcr,
)
from vcr import VCR


@pytest.mark.vcr
@pytest.mark.parametrize("create_adapter", ADAPTER_TEST_FACTORIES, ids=str)
async def test_async(vcr: VCR, create_adapter: AdapterTestFactory) -> None:
    adapter = create_adapter()

    adapter_response = await adapter.execute_async(
        [
            {
                "role": "user",
                "content": "Hi",
            },
            {
                "role": "system",
                "content": "Hi",
            },
            {
                "role": "assistant",
                "content": "Hi",
            },
        ]
    )

    cassette_response = get_response_content_from_vcr(vcr, adapter)

    assert adapter_response.choices[0].message.content == cassette_response
