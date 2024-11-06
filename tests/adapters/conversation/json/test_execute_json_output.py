import pytest

from tests.utils import (
    ADAPTER_TEST_FACTORIES,
    SIMPLE_CONVERSATION_JSON_OUTPUT,
    AdapterTestFactory,
    get_response_content_from_vcr,
)
from vcr import VCR


@pytest.mark.vcr
@pytest.mark.parametrize("create_adapter", ADAPTER_TEST_FACTORIES, ids=str)
async def test_async(vcr: VCR, create_adapter: AdapterTestFactory) -> None:
    adapter = create_adapter()

    if adapter.get_model().supports_json_output is False:
        return

    adapter_response = await adapter.execute_async(
        SIMPLE_CONVERSATION_JSON_OUTPUT,
        response_format={"type": "json_object"},
    )

    cassette_response = get_response_content_from_vcr(vcr, adapter)

    assert adapter_response.choices[0].message.content == cassette_response
