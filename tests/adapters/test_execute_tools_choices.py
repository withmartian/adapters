import pytest

from adapters.adapter_factory import AdapterFactory
from tests.adapters.utils.constants import MODEL_PATHS
from tests.utils import SIMPLE_FUNCTION_CALL_USER_ONLY, get_response_choices_from_vcr


def extract_data(choice):
    if isinstance(choice, dict):
        c = choice["message"]["content"]
        r = choice["message"]["role"]
    else:
        c = choice.message.content
        r = choice.message.role

    return c, r


@pytest.mark.parametrize("model_name", MODEL_PATHS)
@pytest.mark.vcr
def test_sync_execute_tools_choices(vcr, model_name):
    adapter = AdapterFactory.get_adapter_by_path(model_name)

    assert adapter is not None

    if adapter.get_model().supports_tools is False:
        return

    adapter_response = adapter.execute_sync(
        SIMPLE_FUNCTION_CALL_USER_ONLY,
        tool_choice="none",
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
    content = extract_data(adapter_response.choices[0])
    assert choices[0]["message"]["content"] == content


@pytest.mark.parametrize("model_name", MODEL_PATHS)
@pytest.mark.vcr
async def test_async_execute_tools_choices(vcr, model_name):
    adapter = AdapterFactory.get_adapter_by_path(model_name)

    assert adapter is not None

    if adapter.get_model().supports_tools is False:
        return

    adapter_response = await adapter.execute_async(
        SIMPLE_FUNCTION_CALL_USER_ONLY,
        tool_choice="none",
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
    content = extract_data(adapter_response.choices[0])
    assert choices[0]["message"]["content"] == content
