import pytest

from tests.utils import (
    ADAPTER_TEST_FACTORIES,
    SIMPLE_FUNCTION_CALL_USER_ONLY,
    AdapterTestFactory,
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


@pytest.mark.vcr
@pytest.mark.parametrize("create_adapter", ADAPTER_TEST_FACTORIES, ids=str)
async def test_async(vcr: VCR, create_adapter: AdapterTestFactory) -> None:
    adapter = create_adapter()

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
    assert "assistant" == adapter_response.choices[0].message.role
