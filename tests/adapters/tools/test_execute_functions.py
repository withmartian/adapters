import pytest

from adapters.abstract_adapters.base_adapter import BaseAdapter
from tests.utils import (
    TEST_ADAPTERS,
    SIMPLE_FUNCTION_CALL_USER_ONLY,
    get_response_choices_from_vcr,
)
from vcr import VCR


@pytest.mark.vcr
@pytest.mark.parametrize("adapter", TEST_ADAPTERS, ids=lambda adapter: str(adapter))
async def test_async(vcr: VCR, adapter: BaseAdapter) -> None:
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
