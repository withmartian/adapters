import pytest

from tests.utils import (
    ADAPTER_TEST_FACTORIES,
    SIMPLE_FUNCTION_CALL_USER_ONLY,
    AdapterTestFactory,
    get_response_choices_from_vcr,
)
from vcr import VCR


@pytest.mark.vcr
@pytest.mark.parametrize("create_adapter", ADAPTER_TEST_FACTORIES, ids=str)
async def test_async(vcr: VCR, create_adapter: AdapterTestFactory) -> None:
    adapter = create_adapter()

    if adapter.get_model().supports_functions is False:
        return

    adapter_response = await adapter.execute_async(
        SIMPLE_FUNCTION_CALL_USER_ONLY,
        function_call={"name": "generate"},
        functions=[{"description": "Generate random number", "name": "generate"}],
    )
    choices = get_response_choices_from_vcr(vcr, adapter)

    assert adapter_response.choices[0].message.function_call
    assert (
        adapter_response.choices[0].message.function_call.name
        == choices[0]["message"]["function_call"]["name"]
    )
    assert (
        adapter_response.choices[0].message.function_call.arguments
        == choices[0]["message"]["function_call"]["arguments"]
    )
