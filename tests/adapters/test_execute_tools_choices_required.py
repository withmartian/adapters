import pytest

from adapters.adapter_factory import AdapterFactory
from adapters.types import ConversationRole
from tests.adapters.utils.contants import MODEL_PATHS
from tests.utils import SIMPLE_FUNCTION_CALL_USER_ONLY, get_response_choices_from_vcr


def extract_data(choice):
    if isinstance(choice, dict):
        fn = choice["message"]["tool_calls"][0]["function"]["name"]
        fa = choice["message"]["tool_calls"][0]["function"]["arguments"]
        r = choice["message"]["role"]
    else:
        fn = choice.message.tool_calls[0].function.name
        fa = choice.message.tool_calls[0].function.arguments
        r = choice.message.role

    return fn, fa, r


@pytest.mark.parametrize("model_name", MODEL_PATHS)
@pytest.mark.vcr
def test_sync_execute_tools_choices_required(vcr, model_name):
    adapter = AdapterFactory.get_adapter_by_path(model_name)

    assert adapter is not None

    if adapter.get_model().supports_tools is False:
        return

    if adapter.get_model().supports_tool_choice_required is False:
        return

    adapter_response = adapter.execute_sync(
        adapter.convert_to_input(SIMPLE_FUNCTION_CALL_USER_ONLY),
        tool_choice="required",
        tools=[
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
        ],
    )

    choices = get_response_choices_from_vcr(vcr, adapter)
    function_name, function_arguments, role = extract_data(adapter_response.choices[0])
    assert function_name == choices[0]["message"]["tool_calls"][0]["function"]["name"]
    assert (
        function_arguments
        == choices[0]["message"]["tool_calls"][0]["function"]["arguments"]
    )
    assert role == ConversationRole.assistant
    assert adapter_response.cost > 0


@pytest.mark.parametrize("model_name", MODEL_PATHS)
@pytest.mark.vcr
async def test_async_execute_tools_choices_required(vcr, model_name):
    adapter = AdapterFactory.get_adapter_by_path(model_name)

    assert adapter is not None

    if adapter.get_model().supports_tools is False:
        return

    if adapter.get_model().supports_tool_choice_required is False:
        return

    adapter_response = await adapter.execute_async(
        adapter.convert_to_input(SIMPLE_FUNCTION_CALL_USER_ONLY),
        tool_choice="required",
        tools=[
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
        ],
    )

    choices = get_response_choices_from_vcr(vcr, adapter)
    function_name, function_arguments, role = extract_data(adapter_response.choices[0])
    assert function_name == choices[0]["message"]["tool_calls"][0]["function"]["name"]
    assert (
        function_arguments
        == choices[0]["message"]["tool_calls"][0]["function"]["arguments"]
    )
    assert role == ConversationRole.assistant
    assert adapter_response.cost > 0
