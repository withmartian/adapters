import json

import pytest

from adapters import AdapterFactory
from adapters.concrete_adapters.you_com_rag_chat_adapter import YOU_COM_MODEL
from adapters.types import ConversationRole
from tests.utils import SIMPLE_CONVERSATION_YOU_RAG_QUESTION


@pytest.mark.vcr
def test_sync_execute_on_you_com_rag_modal_with_extra_params_ok(vcr):
    adapter = AdapterFactory.get_adapter_by_path(YOU_COM_MODEL.get_path())

    adapter_response = adapter.execute_sync(
        adapter.convert_to_input(SIMPLE_CONVERSATION_YOU_RAG_QUESTION),
        temperature=0,
        max_tokens=10,
        top_p=0.9,
    )

    cassette_response = json.loads(
        vcr.responses[len(vcr.responses) - 1]["body"]["string"]
    )

    assert adapter_response.response.content == cassette_response["answer"]
    assert adapter_response.response.role == ConversationRole.assistant
    assert len(adapter_response.hits) == len(cassette_response["hits"])
    assert adapter_response.latency == cassette_response["latency"]
    assert adapter_response.cost > 0


@pytest.mark.vcr
async def test_async_execute_on_you_com_rag_models_with_extra_params_ok(vcr):
    adapter = AdapterFactory.get_adapter_by_path(YOU_COM_MODEL.get_path())

    adapter_response = await adapter.execute_async(
        adapter.convert_to_input(SIMPLE_CONVERSATION_YOU_RAG_QUESTION),
        temperature=0,
        max_tokens=10,
        top_p=0.9,
        presence_penalty=0.5,
    )
    cassette_response = json.loads(
        vcr.responses[len(vcr.responses) - 1]["body"]["string"]
    )
    assert adapter_response.response.content == cassette_response["answer"]
    assert adapter_response.response.role == ConversationRole.assistant
    assert len(adapter_response.hits) == len(cassette_response["hits"])
    assert adapter_response.latency == cassette_response["latency"]
    assert adapter_response.cost > 0
