import json

import pytest

from adapters import AdapterFactory
from adapters.provider_adapters.anthropic_sdk_chat_provider_adapter import (
    AnthropicSDKChatProviderAdapter,
)
from adapters.types import ConversationRole
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

ANTHROPIC_CHAT_MODELS = [
    formatted
    for model in AnthropicSDKChatProviderAdapter.get_supported_models()
    for formatted in [
        f"{model.name}",
    ]
]


@pytest.mark.parametrize("model_name", ANTHROPIC_CHAT_MODELS)
@pytest.mark.vcr
def test_sync_execute(vcr, model_name):
    adapter = AdapterFactory.get_adapter_by_path(model_name)
    adapter_response = adapter.execute_sync(
        adapter.convert_to_input(SIMPLE_CONVERSATION_USER_ONLY),
    )

    cassette_response = json.loads(
        vcr.responses[len(vcr.responses) - 1]["body"]["string"]
    )["content"][0]["text"]

    assert adapter_response.response.content == cassette_response
    assert adapter_response.response.role == ConversationRole.assistant
    assert adapter_response.cost > 0


@pytest.mark.parametrize("model_name", ANTHROPIC_CHAT_MODELS)
@pytest.mark.vcr
async def test_async_execute(vcr, model_name):
    adapter = AdapterFactory.get_adapter_by_path(model_name)
    adapter_response = await adapter.execute_async(
        adapter.convert_to_input(SIMPLE_CONVERSATION_USER_ONLY),
    )

    cassette_response = json.loads(
        vcr.responses[len(vcr.responses) - 1]["body"]["string"]
    )["content"][0]["text"]

    assert adapter_response.response.content == cassette_response
    assert adapter_response.response.role == ConversationRole.assistant
    assert adapter_response.cost > 0


@pytest.mark.parametrize("model_name", ANTHROPIC_CHAT_MODELS)
@pytest.mark.vcr
def test_sync_repeating_roles(vcr, model_name):
    adapter = AdapterFactory.get_adapter_by_path(model_name)
    adapter_response = adapter.execute_sync(
        adapter.convert_to_input(SIMPLE_CONVERSATION_REPEATING),
    )

    cassette_response = json.loads(
        vcr.responses[len(vcr.responses) - 1]["body"]["string"]
    )["content"][0]["text"]

    assert adapter_response.response.content == cassette_response
    assert adapter_response.response.role == ConversationRole.assistant
    assert adapter_response.cost > 0


@pytest.mark.parametrize("model_name", ANTHROPIC_CHAT_MODELS)
@pytest.mark.vcr
async def test_async_repeating_roles(vcr, model_name):
    adapter = AdapterFactory.get_adapter_by_path(model_name)
    adapter_response = await adapter.execute_async(
        adapter.convert_to_input(SIMPLE_CONVERSATION_REPEATING),
    )

    cassette_response = json.loads(
        vcr.responses[len(vcr.responses) - 1]["body"]["string"]
    )["content"][0]["text"]

    assert adapter_response.response.content == cassette_response
    assert adapter_response.response.role == ConversationRole.assistant
    assert adapter_response.cost > 0


@pytest.mark.parametrize("model_name", ANTHROPIC_CHAT_MODELS)
@pytest.mark.vcr
def test_sync_assistant_first(vcr, model_name):
    adapter = AdapterFactory.get_adapter_by_path(model_name)
    adapter_response = adapter.execute_sync(
        adapter.convert_to_input(SIMPLE_CONVERSATION_ASSISTANT_FIRST),
    )

    cassette_response = json.loads(
        vcr.responses[len(vcr.responses) - 1]["body"]["string"]
    )["content"][0]["text"]

    assert adapter_response.response.content == cassette_response
    assert adapter_response.response.role == ConversationRole.assistant
    assert adapter_response.cost > 0


@pytest.mark.parametrize("model_name", ANTHROPIC_CHAT_MODELS)
@pytest.mark.vcr
async def test_async_assistant_first(vcr, model_name):
    adapter = AdapterFactory.get_adapter_by_path(model_name)
    adapter_response = await adapter.execute_async(
        adapter.convert_to_input(SIMPLE_CONVERSATION_ASSISTANT_FIRST),
    )

    cassette_response = json.loads(
        vcr.responses[len(vcr.responses) - 1]["body"]["string"]
    )["content"][0]["text"]

    assert adapter_response.response.content == cassette_response
    assert adapter_response.response.role == ConversationRole.assistant
    assert adapter_response.cost > 0


@pytest.mark.parametrize("model_name", ANTHROPIC_CHAT_MODELS)
@pytest.mark.vcr
def test_sync_multiple_system(vcr, model_name):
    adapter = AdapterFactory.get_adapter_by_path(model_name)
    adapter_response = adapter.execute_sync(
        adapter.convert_to_input(SIMPLE_CONVERSATION_MULTIPLE_SYSTEM),
    )

    cassette_response = json.loads(
        vcr.responses[len(vcr.responses) - 1]["body"]["string"]
    )["content"][0]["text"]

    assert adapter_response.response.content == cassette_response
    assert adapter_response.response.role == ConversationRole.assistant
    assert adapter_response.cost > 0


@pytest.mark.parametrize("model_name", ANTHROPIC_CHAT_MODELS)
@pytest.mark.vcr
async def test_async_multiple_system(vcr, model_name):
    adapter = AdapterFactory.get_adapter_by_path(model_name)
    adapter_response = await adapter.execute_async(
        adapter.convert_to_input(SIMPLE_CONVERSATION_MULTIPLE_SYSTEM),
    )

    cassette_response = json.loads(
        vcr.responses[len(vcr.responses) - 1]["body"]["string"]
    )["content"][0]["text"]

    assert adapter_response.response.content == cassette_response
    assert adapter_response.response.role == ConversationRole.assistant
    assert adapter_response.cost > 0


@pytest.mark.parametrize("model_name", ANTHROPIC_CHAT_MODELS)
@pytest.mark.vcr
def test_sync_empty_content(vcr, model_name):
    adapter = AdapterFactory.get_adapter_by_path(model_name)
    adapter_response = adapter.execute_sync(
        adapter.convert_to_input(SIMPLE_CONVERSATION_EMPTY_CONTENT),
    )

    cassette_response = json.loads(
        vcr.responses[len(vcr.responses) - 1]["body"]["string"]
    )["content"][0]["text"]

    assert adapter_response.response.content == cassette_response
    assert adapter_response.response.role == ConversationRole.assistant
    assert adapter_response.cost > 0


@pytest.mark.parametrize("model_name", ANTHROPIC_CHAT_MODELS)
@pytest.mark.vcr
async def test_async_empty_content(vcr, model_name):
    adapter = AdapterFactory.get_adapter_by_path(model_name)
    adapter_response = await adapter.execute_async(
        adapter.convert_to_input(SIMPLE_CONVERSATION_EMPTY_CONTENT),
    )

    cassette_response = json.loads(
        vcr.responses[len(vcr.responses) - 1]["body"]["string"]
    )["content"][0]["text"]

    assert adapter_response.response.content == cassette_response
    assert adapter_response.response.role == ConversationRole.assistant
    assert adapter_response.cost > 0


@pytest.mark.parametrize("model_name", ANTHROPIC_CHAT_MODELS)
@pytest.mark.vcr
def test_sync_trailing_whitespace(vcr, model_name):
    adapter = AdapterFactory.get_adapter_by_path(model_name)
    adapter_response = adapter.execute_sync(
        adapter.convert_to_input(SIMPLE_CONVERSATION_TRAILING_WHITESPACE),
    )

    cassette_response = json.loads(
        vcr.responses[len(vcr.responses) - 1]["body"]["string"]
    )["content"][0]["text"]

    assert adapter_response.response.content == cassette_response
    assert adapter_response.response.role == ConversationRole.assistant
    assert adapter_response.cost > 0


@pytest.mark.parametrize("model_name", ANTHROPIC_CHAT_MODELS)
@pytest.mark.vcr
async def test_async_trailing_whitespace(vcr, model_name):
    adapter = AdapterFactory.get_adapter_by_path(model_name)
    adapter_response = await adapter.execute_async(
        adapter.convert_to_input(SIMPLE_CONVERSATION_TRAILING_WHITESPACE),
    )

    cassette_response = json.loads(
        vcr.responses[len(vcr.responses) - 1]["body"]["string"]
    )["content"][0]["text"]

    assert adapter_response.response.content == cassette_response
    assert adapter_response.response.role == ConversationRole.assistant
    assert adapter_response.cost > 0


@pytest.mark.parametrize("model_name", ANTHROPIC_CHAT_MODELS)
@pytest.mark.vcr
def test_sync_system_assistant(vcr, model_name):
    adapter = AdapterFactory.get_adapter_by_path(model_name)
    adapter_response = adapter.execute_sync(
        adapter.convert_to_input(SIMPLE_CONVERSATION_ASSISTANT_SYSTEM),
    )

    cassette_response = json.loads(
        vcr.responses[len(vcr.responses) - 1]["body"]["string"]
    )["content"][0]["text"]

    assert adapter_response.response.content == cassette_response
    assert adapter_response.response.role == ConversationRole.assistant
    assert adapter_response.cost > 0


@pytest.mark.parametrize("model_name", ANTHROPIC_CHAT_MODELS)
@pytest.mark.vcr
async def test_async_system_assistant(vcr, model_name):
    adapter = AdapterFactory.get_adapter_by_path(model_name)
    adapter_response = await adapter.execute_async(
        adapter.convert_to_input(SIMPLE_CONVERSATION_ASSISTANT_SYSTEM),
    )

    cassette_response = json.loads(
        vcr.responses[len(vcr.responses) - 1]["body"]["string"]
    )["content"][0]["text"]

    assert adapter_response.response.content == cassette_response
    assert adapter_response.response.role == ConversationRole.assistant
    assert adapter_response.cost > 0


@pytest.mark.parametrize("model_name", ANTHROPIC_CHAT_MODELS)
@pytest.mark.vcr
def test_sync_execute_json_content(vcr, model_name):
    adapter = AdapterFactory.get_adapter_by_path(model_name)

    if adapter.get_model().supports_json_content is False:
        return

    adapter_response = adapter.execute_sync(
        adapter.convert_to_input(SIMPLE_CONVERSATION_JSON_CONTENT),
    )

    cassette_response = json.loads(
        vcr.responses[len(vcr.responses) - 1]["body"]["string"]
    )["content"][0]["text"]

    assert adapter_response.response.content == cassette_response
    assert adapter_response.response.role == ConversationRole.assistant
    assert adapter_response.cost > 0


@pytest.mark.parametrize("model_name", ANTHROPIC_CHAT_MODELS)
@pytest.mark.vcr
async def test_async_execute_json_content(vcr, model_name):
    adapter = AdapterFactory.get_adapter_by_path(model_name)

    if adapter.get_model().supports_json_content is False:
        return

    adapter_response = await adapter.execute_async(
        adapter.convert_to_input(SIMPLE_CONVERSATION_JSON_CONTENT),
    )

    cassette_response = json.loads(
        vcr.responses[len(vcr.responses) - 1]["body"]["string"]
    )["content"][0]["text"]

    assert adapter_response.response.content == cassette_response
    assert adapter_response.response.role == ConversationRole.assistant
    assert adapter_response.cost > 0


@pytest.mark.parametrize("model_name", ANTHROPIC_CHAT_MODELS)
@pytest.mark.vcr
async def test_async_execute_streaming(
    vcr, model_name  # pylint: disable=unused-argument
):
    adapter = AdapterFactory.get_adapter_by_path(model_name)

    if not adapter.get_model().supports_streaming:
        return

    response = await adapter.execute_async(
        adapter.convert_to_input(SIMPLE_CONVERSATION_USER_ONLY),
        stream=True,
    )

    async for chunk in response.response:
        assert chunk


@pytest.mark.parametrize("model_name", ANTHROPIC_CHAT_MODELS)
@pytest.mark.vcr
def test_sync_execute_streaming(vcr, model_name):  # pylint: disable=unused-argument
    adapter = AdapterFactory.get_adapter_by_path(model_name)

    if not adapter.get_model().supports_streaming:
        return

    response = adapter.execute_sync(
        adapter.convert_to_input(SIMPLE_CONVERSATION_USER_ONLY),
        stream=True,
    )

    for chunk in response.response:
        assert chunk
