import pytest

from adapters.abstract_adapters.base_adapter import BaseAdapter
from tests.utils import (
    TEST_ADAPTERS,
    SIMPLE_FUNCTION_CALL_USER_ONLY,
    get_response_choices_from_vcr,
)
from vcr import VCR


tools = [
    {
        "type": "function",
        "function": {
            "description": "Generate random number",
            "name": "generate",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Random number like 5, 4, 3, 10, 11",
                    },
                },
                "required": ["prompt"],
            },
        },
    }
]


@pytest.mark.parametrize("adapter", TEST_ADAPTERS)
@pytest.mark.vcr
async def test_async(vcr: VCR, adapter: BaseAdapter) -> None:
    if adapter.get_model().supports_tools is False:
        return

    adapter_response = await adapter.execute_async(
        SIMPLE_FUNCTION_CALL_USER_ONLY, tool_choice="none", tools=tools
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
