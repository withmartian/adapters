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
        or adapter.get_model().supports_tools_choice is False
        or adapter.get_model().supports_tools_choice_required is False
    ):
        return

    adapter_response = await adapter.execute_async(
        SIMPLE_FUNCTION_CALL_USER_ONLY, tools=SIMPLE_GENERATE_TOOLS, tool_choice="none"
    )

    choices = get_response_choices_from_vcr(vcr, adapter)

    assert (
        choices[0]["message"].get("content", None)
        == adapter_response.choices[0].message.content
    )
    assert (
        choices[0]["message"].get("role", None)
        == adapter_response.choices[0].message.role
    )
