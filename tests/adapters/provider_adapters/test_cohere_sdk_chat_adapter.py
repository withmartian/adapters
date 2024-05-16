import json
from typing import List

from pytest import mark

from adapters import AdapterFactory
from adapters.provider_adapters.cohere_sdk_chat_provider_adapter import (
    CohereSDKChatProviderAdapter,
)
from adapters.types import ConversationRole, Model
from tests.utils import (
    SIMPLE_CONVERSATION_ASSISTANT_FIRST,
    SIMPLE_CONVERSATION_ASSISTANT_SYSTEM,
    SIMPLE_CONVERSATION_EMPTY_CONTENT,
    SIMPLE_CONVERSATION_JSON_CONTENT,
    SIMPLE_CONVERSATION_MULTIPLE_SYSTEM,
    SIMPLE_CONVERSATION_REPEATING,
    SIMPLE_CONVERSATION_TRAILING_WHITESPACE,
    SIMPLE_CONVERSATION_USER_ONLY,
)

MODELS: List[Model] = [
    *CohereSDKChatProviderAdapter.get_supported_models(),
]

COHERE_CHAT_MODELS = [model.get_path() for model in MODELS]


@mark.parametrize("model_name", COHERE_CHAT_MODELS)
@mark.vcr
def test_sync_execute(vcr, model_name: str) -> None:
    adapter = AdapterFactory.get_adapter_by_path(model_name)
    assert isinstance(adapter, CohereSDKChatProviderAdapter)
    adapter_response = adapter.execute_sync(
        adapter.convert_to_input(SIMPLE_CONVERSATION_USER_ONLY),
    )

    cassette_response = json.loads(vcr.responses[-1]["body"]["string"])["text"]

    assert adapter_response.response.content == cassette_response
    assert adapter_response.response.role == ConversationRole.assistant
    assert adapter_response.cost > 0


@mark.parametrize("model_name", COHERE_CHAT_MODELS)
@mark.vcr
async def test_async_execute(vcr, model_name: str) -> None:
    adapter = AdapterFactory.get_adapter_by_path(model_name)
    assert isinstance(adapter, CohereSDKChatProviderAdapter)
    adapter_response = await adapter.execute_async(
        adapter.convert_to_input(SIMPLE_CONVERSATION_USER_ONLY),
    )

    cassette_response = json.loads(vcr.responses[-1]["body"]["string"])["text"]

    assert adapter_response.response.content == cassette_response
    assert adapter_response.response.role == ConversationRole.assistant
    assert adapter_response.cost > 0


@mark.parametrize("model_name", COHERE_CHAT_MODELS)
@mark.vcr
def test_sync_repeating_roles(vcr, model_name: str) -> None:
    adapter = AdapterFactory.get_adapter_by_path(model_name)
    assert isinstance(adapter, CohereSDKChatProviderAdapter)
    adapter_response = adapter.execute_sync(
        adapter.convert_to_input(SIMPLE_CONVERSATION_REPEATING),
    )

    cassette_response = json.loads(vcr.responses[-1]["body"]["string"])["text"]

    assert adapter_response.response.content == cassette_response
    assert adapter_response.response.role == ConversationRole.assistant
    assert adapter_response.cost > 0


@mark.parametrize("model_name", COHERE_CHAT_MODELS)
@mark.vcr
async def test_async_repeating_roles(vcr, model_name: str) -> None:
    adapter = AdapterFactory.get_adapter_by_path(model_name)
    assert isinstance(adapter, CohereSDKChatProviderAdapter)
    adapter_response = await adapter.execute_async(
        adapter.convert_to_input(SIMPLE_CONVERSATION_REPEATING),
    )

    cassette_response = json.loads(vcr.responses[-1]["body"]["string"])["text"]

    assert adapter_response.response.content == cassette_response
    assert adapter_response.response.role == ConversationRole.assistant
    assert adapter_response.cost > 0


@mark.parametrize("model_name", COHERE_CHAT_MODELS)
@mark.vcr
def test_sync_assistant_first(vcr, model_name: str) -> None:
    adapter = AdapterFactory.get_adapter_by_path(model_name)
    assert isinstance(adapter, CohereSDKChatProviderAdapter)
    adapter_response = adapter.execute_sync(
        adapter.convert_to_input(SIMPLE_CONVERSATION_ASSISTANT_FIRST),
    )

    cassette_response = json.loads(vcr.responses[-1]["body"]["string"])["text"]

    assert adapter_response.response.content == cassette_response
    assert adapter_response.response.role == ConversationRole.assistant
    assert adapter_response.cost > 0


@mark.parametrize("model_name", COHERE_CHAT_MODELS)
@mark.vcr
async def test_async_assistant_first(vcr, model_name: str) -> None:
    adapter = AdapterFactory.get_adapter_by_path(model_name)
    assert isinstance(adapter, CohereSDKChatProviderAdapter)
    adapter_response = await adapter.execute_async(
        adapter.convert_to_input(SIMPLE_CONVERSATION_ASSISTANT_FIRST),
    )

    cassette_response = json.loads(vcr.responses[-1]["body"]["string"])["text"]

    assert adapter_response.response.content == cassette_response
    assert adapter_response.response.role == ConversationRole.assistant
    assert adapter_response.cost > 0


@mark.parametrize("model_name", COHERE_CHAT_MODELS)
@mark.vcr
def test_sync_multiple_system(vcr, model_name: str) -> None:
    adapter = AdapterFactory.get_adapter_by_path(model_name)
    assert isinstance(adapter, CohereSDKChatProviderAdapter)
    adapter_response = adapter.execute_sync(
        adapter.convert_to_input(SIMPLE_CONVERSATION_MULTIPLE_SYSTEM),
    )

    cassette_response = json.loads(vcr.responses[-1]["body"]["string"])["text"]

    assert adapter_response.response.content == cassette_response
    assert adapter_response.response.role == ConversationRole.assistant
    assert adapter_response.cost > 0


@mark.parametrize("model_name", COHERE_CHAT_MODELS)
@mark.vcr
async def test_async_multiple_system(vcr, model_name: str) -> None:
    adapter = AdapterFactory.get_adapter_by_path(model_name)
    assert isinstance(adapter, CohereSDKChatProviderAdapter)
    adapter_response = await adapter.execute_async(
        adapter.convert_to_input(SIMPLE_CONVERSATION_MULTIPLE_SYSTEM),
    )

    cassette_response = json.loads(vcr.responses[-1]["body"]["string"])["text"]

    assert adapter_response.response.content == cassette_response
    assert adapter_response.response.role == ConversationRole.assistant
    assert adapter_response.cost > 0


@mark.parametrize("model_name", COHERE_CHAT_MODELS)
@mark.vcr
def test_sync_empty_content(vcr, model_name: str) -> None:
    adapter = AdapterFactory.get_adapter_by_path(model_name)
    assert isinstance(adapter, CohereSDKChatProviderAdapter)
    adapter_response = adapter.execute_sync(
        adapter.convert_to_input(SIMPLE_CONVERSATION_EMPTY_CONTENT),
    )

    cassette_response = json.loads(vcr.responses[-1]["body"]["string"])["text"]

    assert adapter_response.response.content == cassette_response
    assert adapter_response.response.role == ConversationRole.assistant
    assert adapter_response.cost > 0


@mark.parametrize("model_name", COHERE_CHAT_MODELS)
@mark.vcr
async def test_async_empty_content(vcr, model_name: str) -> None:
    adapter = AdapterFactory.get_adapter_by_path(model_name)
    assert isinstance(adapter, CohereSDKChatProviderAdapter)
    adapter_response = await adapter.execute_async(
        adapter.convert_to_input(SIMPLE_CONVERSATION_EMPTY_CONTENT),
    )

    cassette_response = json.loads(vcr.responses[-1]["body"]["string"])["text"]

    assert adapter_response.response.content == cassette_response
    assert adapter_response.response.role == ConversationRole.assistant
    assert adapter_response.cost > 0


@mark.parametrize("model_name", COHERE_CHAT_MODELS)
@mark.vcr
def test_sync_trailing_whitespace(vcr, model_name: str) -> None:
    adapter = AdapterFactory.get_adapter_by_path(model_name)
    assert isinstance(adapter, CohereSDKChatProviderAdapter)
    adapter_response = adapter.execute_sync(
        adapter.convert_to_input(SIMPLE_CONVERSATION_TRAILING_WHITESPACE),
    )

    cassette_response = json.loads(vcr.responses[-1]["body"]["string"])["text"]

    assert adapter_response.response.content == cassette_response
    assert adapter_response.response.role == ConversationRole.assistant
    assert adapter_response.cost > 0


@mark.parametrize("model_name", COHERE_CHAT_MODELS)
@mark.vcr
async def test_async_trailing_whitespace(vcr, model_name: str) -> None:
    adapter = AdapterFactory.get_adapter_by_path(model_name)
    assert isinstance(adapter, CohereSDKChatProviderAdapter)
    adapter_response = await adapter.execute_async(
        adapter.convert_to_input(SIMPLE_CONVERSATION_TRAILING_WHITESPACE),
    )

    cassette_response = json.loads(vcr.responses[-1]["body"]["string"])["text"]

    assert adapter_response.response.content == cassette_response
    assert adapter_response.response.role == ConversationRole.assistant
    assert adapter_response.cost > 0


@mark.parametrize("model_name", COHERE_CHAT_MODELS)
@mark.vcr
def test_sync_system_assistant(vcr, model_name: str) -> None:
    adapter = AdapterFactory.get_adapter_by_path(model_name)
    assert isinstance(adapter, CohereSDKChatProviderAdapter)
    adapter_response = adapter.execute_sync(
        adapter.convert_to_input(SIMPLE_CONVERSATION_ASSISTANT_SYSTEM),
    )

    cassette_response = json.loads(vcr.responses[-1]["body"]["string"])["text"]

    assert adapter_response.response.content == cassette_response
    assert adapter_response.response.role == ConversationRole.assistant
    assert adapter_response.cost > 0


@mark.parametrize("model_name", COHERE_CHAT_MODELS)
@mark.vcr
async def test_async_system_assistant(vcr, model_name: str) -> None:
    adapter = AdapterFactory.get_adapter_by_path(model_name)
    assert isinstance(adapter, CohereSDKChatProviderAdapter)
    adapter_response = await adapter.execute_async(
        adapter.convert_to_input(SIMPLE_CONVERSATION_ASSISTANT_SYSTEM),
    )

    cassette_response = json.loads(vcr.responses[-1]["body"]["string"])["text"]

    assert adapter_response.response.content == cassette_response
    assert adapter_response.response.role == ConversationRole.assistant
    assert adapter_response.cost > 0


@mark.parametrize("model_name", COHERE_CHAT_MODELS)
@mark.vcr
def test_sync_execute_json_content(vcr, model_name):
    adapter = AdapterFactory.get_adapter_by_path(model_name)
    assert isinstance(adapter, CohereSDKChatProviderAdapter)

    if adapter.get_model().supports_json_content is False:
        return

    adapter_response = adapter.execute_sync(
        adapter.convert_to_input(SIMPLE_CONVERSATION_JSON_CONTENT),
    )

    cassette_response = json.loads(vcr.responses[-1]["body"]["string"])["text"]

    assert adapter_response.response.content == cassette_response
    assert adapter_response.response.role == ConversationRole.assistant
    assert adapter_response.cost > 0


@mark.parametrize("model_name", COHERE_CHAT_MODELS)
@mark.vcr
async def test_async_execute_json_content(vcr, model_name):
    adapter = AdapterFactory.get_adapter_by_path(model_name)
    assert isinstance(adapter, CohereSDKChatProviderAdapter)

    if adapter.get_model().supports_json_content is False:
        return

    adapter_response = await adapter.execute_async(
        adapter.convert_to_input(SIMPLE_CONVERSATION_JSON_CONTENT),
    )

    cassette_response = json.loads(vcr.responses[-1]["body"]["string"])["text"]

    assert adapter_response.response.content == cassette_response
    assert adapter_response.response.role == ConversationRole.assistant
    assert adapter_response.cost > 0


@mark.parametrize("model_name", COHERE_CHAT_MODELS)
@mark.vcr
def test_sync_execute_streaming(
    vcr, model_name: str  # pylint: disable=unused-argument
) -> None:
    adapter = AdapterFactory.get_adapter_by_path(model_name)
    assert isinstance(adapter, CohereSDKChatProviderAdapter)

    if not adapter.get_model().supports_streaming:
        return

    adapter_response = adapter.execute_sync(
        adapter.convert_to_input(SIMPLE_CONVERSATION_USER_ONLY),
        stream=True,
    )

    chunks = [
        json.loads(data_chunk[6:].strip()) for data_chunk in adapter_response.response
    ]

    response = "".join(
        [
            chunk["choices"][0]["delta"]["content"]
            for chunk in chunks
            if chunk["choices"][0]["delta"]["content"]
        ]
    )

    assert len(response) > 0


@mark.parametrize("model_name", COHERE_CHAT_MODELS)
@mark.vcr
async def test_async_execute_streaming(
    vcr, model_name: str  # pylint: disable=unused-argument
) -> None:
    adapter = AdapterFactory.get_adapter_by_path(model_name)
    assert isinstance(adapter, CohereSDKChatProviderAdapter)

    if not adapter.get_model().supports_streaming:
        return

    adapter_response = await adapter.execute_async(
        adapter.convert_to_input(SIMPLE_CONVERSATION_USER_ONLY),
        stream=True,
    )

    chunks = [
        json.loads(data_chunk[6:].strip())
        async for data_chunk in adapter_response.response
    ]

    response = "".join(
        [
            chunk["choices"][0]["delta"]["content"]
            for chunk in chunks
            if chunk["choices"][0]["delta"]["content"]
        ]
    )

    assert len(response) > 0
