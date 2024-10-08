import pytest

from adapters.adapter_factory import AdapterFactory
from tests.adapters.utils.constants import MODEL_PATHS
from tests.utils import SIMPLE_FUNCTION_CALL_USER_ONLY, get_response_choices_from_vcr


@pytest.mark.parametrize("model_path", MODEL_PATHS)
@pytest.mark.vcr
def test_sync_execute_function_calls(vcr, model_path: str):
    adapter = AdapterFactory.get_adapter_by_path(model_path)

    assert adapter is not None

    if adapter.get_model().supports_functions is False:
        return

    adapter_response = adapter.execute_sync(
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


@pytest.mark.parametrize("model_path", MODEL_PATHS)
@pytest.mark.vcr
async def test_async_execute_function_calls(vcr, model_path: str):
    adapter = AdapterFactory.get_adapter_by_path(model_path)

    assert adapter is not None

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
