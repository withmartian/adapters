import pytest

from adapters.adapter_factory import AdapterFactory
from adapters.types import ConversationRole
from tests.adapters.utils.contants import MODEL_PATHS, MODEL_PATHS_ASYNC
from tests.utils import SIMPLE_CONVERSATION_EMPTY_CONTENT, get_response_content_from_vcr


@pytest.mark.parametrize("model_path", MODEL_PATHS)
@pytest.mark.vcr
def test_sync(vcr, model_path: str):
    adapter = AdapterFactory.get_adapter_by_path(model_path)

    assert adapter is not None

    adapter_response = adapter.execute_sync(
        adapter.convert_to_input(SIMPLE_CONVERSATION_EMPTY_CONTENT),
    )

    cassette_response = get_response_content_from_vcr(vcr, adapter)

    assert adapter_response.response.content == cassette_response
    assert adapter_response.response.role == ConversationRole.assistant

    finish_reason = getattr(adapter_response.choices[0], "finish_reason", None)  # type: ignore
    assert finish_reason in ["stop", "eos", "length", None]


@pytest.mark.parametrize("model_path", MODEL_PATHS_ASYNC)
@pytest.mark.vcr
async def test_async(vcr, model_path: str):
    adapter = AdapterFactory.get_adapter_by_path(model_path)

    assert adapter is not None

    adapter_response = await adapter.execute_async(
        adapter.convert_to_input(SIMPLE_CONVERSATION_EMPTY_CONTENT),
    )

    cassette_response = get_response_content_from_vcr(vcr, adapter)

    assert adapter_response.response.content == cassette_response
    assert adapter_response.response.role == ConversationRole.assistant

    finish_reason = getattr(adapter_response.choices[0], "finish_reason", None)  # type: ignore
    assert finish_reason in ["stop", "eos", "length", None]
