import pytest

from tests.utils import (
    ADAPTER_TEST_FACTORIES,
    SIMPLE_FUNCTION_CALL_USER_ONLY,
    SIMPLE_GENERATE_TOOLS,
    AdapterTestFactory,
    get_response_choices_from_vcr,
)
from vcr import VCR


@pytest.mark.vcr
@pytest.mark.parametrize("create_adapter", ADAPTER_TEST_FACTORIES, ids=str)
async def test_async(vcr: VCR, create_adapter: AdapterTestFactory) -> None:
    adapter = create_adapter()

    if (
        adapter.get_model().supports_tools is False
        or adapter.get_model().supports_tool_choice is False
        or adapter.get_model().supports_tool_choice_required is False
    ):
        return

    adapter_response = await adapter.execute_async(
        SIMPLE_FUNCTION_CALL_USER_ONLY,
        tools=SIMPLE_GENERATE_TOOLS,
        tool_choice="required",
    )

    choices = get_response_choices_from_vcr(vcr, adapter)

    assert adapter_response.choices[0].message.tool_calls
    assert (
        adapter_response.choices[0].message.tool_calls[0].function.name
        == choices[0]["message"]["tool_calls"][0]["function"]["name"]
    )
    assert (
        adapter_response.choices[0].message.tool_calls[0].function.arguments
        == choices[0]["message"]["tool_calls"][0]["function"]["arguments"]
    )
