import pytest

from adapters.adapter_factory import AdapterFactory
from adapters.types import ConversationRole
from tests.adapters.utils.contants import MODEL_PATHS, N_PARAM
from tests.utils import SIMPLE_CONVERSATION_USER_ONLY, get_choices_from_vcr


@pytest.mark.parametrize("model_path", MODEL_PATHS)
@pytest.mark.vcr
def test_sync_execute_n(vcr, model_path: str):
    adapter = AdapterFactory.get_adapter_by_path(model_path)

    assert adapter is not None

    if adapter.get_model().supports_n is False:
        return

    adapter_response = adapter.execute_sync(
        adapter.convert_to_input(SIMPLE_CONVERSATION_USER_ONLY), n=N_PARAM
    )

    choices = get_choices_from_vcr(vcr, adapter)

    assert adapter_response.response.content == choices[0]["message"]["content"]
    assert adapter_response.response.role == ConversationRole.assistant
    assert len(choices) == N_PARAM


@pytest.mark.parametrize("model_path", MODEL_PATHS)
@pytest.mark.vcr
async def test_async_execute_n(vcr, model_path: str):
    adapter = AdapterFactory.get_adapter_by_path(model_path)

    assert adapter is not None

    if adapter.get_model().supports_n is False:
        return

    adapter_response = await adapter.execute_async(
        adapter.convert_to_input(SIMPLE_CONVERSATION_USER_ONLY), n=N_PARAM
    )

    choices = get_choices_from_vcr(vcr, adapter)

    assert adapter_response.response.content == choices[0]["message"]["content"]
    assert adapter_response.response.role == ConversationRole.assistant
    assert len(choices) == N_PARAM
