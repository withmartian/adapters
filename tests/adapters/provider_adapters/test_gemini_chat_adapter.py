import json
from collections.abc import Iterator
from unittest.mock import AsyncMock, _patch, patch

from google.generativeai.protos import CountTokensResponse
from google.generativeai.protos import (
    GenerateContentResponse as GenerateContentResponseProtos,
)
from google.generativeai.types.generation_types import (
    AsyncGenerateContentResponse,
    GenerateContentResponse,
)
from pytest import fixture, mark

from adapters import AdapterFactory
from adapters.provider_adapters.gemini_sdk_chat_provider_adapter import (
    GeminiSDKChatProviderAdapter,
)
from adapters.types import (
    ConversationRole,
    Cost,
    Model,
    OpenAIChatAdapterResponse,
    Turn,
)
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

MODELS: list[Model] = [
    *GeminiSDKChatProviderAdapter.get_supported_models(),
]

MODEL_NAMES = [f"{model.name}" for model in MODELS]


ASYNC_EXECUTE_EXPECTED_RESPONSES = {
    "gemini-1.0-pro": OpenAIChatAdapterResponse(
        response=Turn(
            role=ConversationRole.assistant,
            content="Hi there! How can I help you today?!",
        ),
        token_counts=Cost(prompt=10.0, completion=20.0, request=0.0),
        # Calculate cost based on prompt and completion tokens
        cost=MODELS[0].cost.prompt * 10 + MODELS[0].cost.completion * 20,
        finish_reason=None,
        choices=[
            {
                "message": {
                    "role": ConversationRole.assistant,
                    "content": "Hi there! How can I help you today?!",
                },
                "finish_reason": "stop",
            },
        ],
    ),
    "gemini-1.5-pro-latest": OpenAIChatAdapterResponse(
        response=Turn(
            role=ConversationRole.assistant,
            content="Hi there! How can I help you today?!",
        ),
        token_counts=Cost(prompt=10.0, completion=20.0, request=0.0),
        # Calculate cost based on prompt and completion tokens
        cost=MODELS[1].cost.prompt * 10 + MODELS[1].cost.completion * 20,
        finish_reason=None,
        choices=[
            {
                "message": {
                    "role": ConversationRole.assistant,
                    "content": "Hi there! How can I help you today?!",
                },
                "finish_reason": "stop",
            },
        ],
    ),
}


ASYNC_EXECUTE_RESPONSE = AsyncGenerateContentResponse(
    done=True,
    iterator=None,
    result=GenerateContentResponseProtos(
        {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "Hi there! How can I help you today?!"}],
                        "role": "model",
                    },
                    "finish_reason": "STOP",
                    "index": 0,
                    "safety_ratings": [
                        {
                            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                            "probability": "NEGLIGIBLE",
                        },
                        {
                            "category": "HARM_CATEGORY_HATE_SPEECH",
                            "probability": "NEGLIGIBLE",
                        },
                        {
                            "category": "HARM_CATEGORY_HARASSMENT",
                            "probability": "NEGLIGIBLE",
                        },
                        {
                            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                            "probability": "NEGLIGIBLE",
                        },
                    ],
                }
            ],
            "usage_metadata": {
                "prompt_token_count": 10,
                "candidates_token_count": 20,
                "total_token_count": 30,
            },
        }
    ),
)

ASYNC_STREAM_RESPONSE = GenerateContentResponse(
    done=True,
    iterator=None,
    result=GenerateContentResponseProtos(
        {
            "candidates": [
                {
                    "content": {"parts": [{"text": "Hi"}], "role": "model"},
                    "finish_reason": "STOP",
                    "index": 0,
                    "safety_ratings": [
                        {
                            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                            "probability": "NEGLIGIBLE",
                        },
                        {
                            "category": "HARM_CATEGORY_HATE_SPEECH",
                            "probability": "NEGLIGIBLE",
                        },
                        {
                            "category": "HARM_CATEGORY_HARASSMENT",
                            "probability": "NEGLIGIBLE",
                        },
                        {
                            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                            "probability": "NEGLIGIBLE",
                        },
                    ],
                }
            ],
            "usage_metadata": {
                "prompt_token_count": 5,
                "candidates_token_count": 16,
                "total_token_count": 21,
            },
        }
    ),
)


def mock_execute_async(
    mock_return_value: AsyncGenerateContentResponse,
) -> tuple[_patch, AsyncMock, AsyncMock]:
    patcher_start_chat = patch(
        "adapters.provider_adapters.gemini_sdk_chat_provider_adapter.genai.GenerativeModel.start_chat"
    )
    mock_start_chat = patcher_start_chat.start()

    mock_chat = AsyncMock()
    mock_chat.send_message_async.return_value = mock_return_value

    mock_start_chat.return_value = mock_chat

    patcher_count_tokens_async = patch(
        "adapters.provider_adapters.gemini_sdk_chat_provider_adapter.genai.GenerativeModel.count_tokens_async"
    )
    mock_count_tokens_async = patcher_count_tokens_async.start()
    # Mock prompt and completion tokens.
    mock_count_tokens_async.side_effect = [
        CountTokensResponse(total_tokens=10),
        CountTokensResponse(total_tokens=20),
    ]

    return patcher_start_chat, mock_chat, mock_count_tokens_async


@fixture
def mock_execute_async_responses() -> Iterator[None]:
    patcher, _, _ = mock_execute_async(ASYNC_EXECUTE_RESPONSE)
    yield
    patcher.stop()


@mark.parametrize("model_name", MODEL_NAMES)
@mark.vcr
def test_sync_execute(vcr, model_name: str) -> None:
    adapter = AdapterFactory.get_adapter_by_path(model_name)
    assert isinstance(adapter, GeminiSDKChatProviderAdapter)

    adapter_response = adapter.execute_sync(
        adapter.convert_to_input(SIMPLE_CONVERSATION_USER_ONLY)
    )

    cassette_response = json.loads(vcr.responses[0]["body"]["string"])

    assert (
        adapter_response.response.content
        == cassette_response["candidates"][0]["content"]["parts"][0]["text"]
    )
    assert adapter_response.response.role == ConversationRole.assistant
    assert adapter_response.cost > 0


@mark.parametrize("model_name", MODEL_NAMES)
async def test_async_execute(
    mock_execute_async_responses: AsyncMock,  # pylint: disable=unused-argument,redefined-outer-name
    model_name: str,
) -> None:
    adapter = AdapterFactory.get_adapter_by_path(model_name)
    assert isinstance(adapter, GeminiSDKChatProviderAdapter)
    adapter_response = await adapter.execute_async(
        adapter.convert_to_input(SIMPLE_CONVERSATION_USER_ONLY)
    )

    assert adapter_response == ASYNC_EXECUTE_EXPECTED_RESPONSES[model_name]


@mark.parametrize("model_name", MODEL_NAMES)
@mark.vcr
def test_sync_repeating_roles(vcr, model_name: str) -> None:
    adapter = AdapterFactory.get_adapter_by_path(model_name)
    assert isinstance(adapter, GeminiSDKChatProviderAdapter)
    adapter_response = adapter.execute_sync(
        adapter.convert_to_input(SIMPLE_CONVERSATION_REPEATING),
    )

    cassette_response = json.loads(vcr.responses[0]["body"]["string"])

    assert (
        adapter_response.response.content
        == cassette_response["candidates"][0]["content"]["parts"][0]["text"]
    )
    assert adapter_response.response.role == ConversationRole.assistant
    assert adapter_response.cost > 0


@mark.parametrize("model_name", MODEL_NAMES)
async def test_async_repeating_roles(
    mock_execute_async_responses: AsyncMock,  # pylint: disable=unused-argument,redefined-outer-name
    model_name: str,
) -> None:
    adapter = AdapterFactory.get_adapter_by_path(model_name)
    assert isinstance(adapter, GeminiSDKChatProviderAdapter)
    adapter_response = await adapter.execute_async(
        adapter.convert_to_input(SIMPLE_CONVERSATION_REPEATING)
    )

    assert adapter_response == ASYNC_EXECUTE_EXPECTED_RESPONSES[model_name]


@mark.parametrize("model_name", MODEL_NAMES)
@mark.vcr
def test_sync_assistant_first(vcr, model_name: str) -> None:
    adapter = AdapterFactory.get_adapter_by_path(model_name)
    assert isinstance(adapter, GeminiSDKChatProviderAdapter)
    adapter_response = adapter.execute_sync(
        adapter.convert_to_input(SIMPLE_CONVERSATION_ASSISTANT_FIRST),
    )

    cassette_response = json.loads(vcr.responses[0]["body"]["string"])

    assert (
        adapter_response.response.content
        == cassette_response["candidates"][0]["content"]["parts"][0]["text"]
    )
    assert adapter_response.response.role == ConversationRole.assistant
    assert adapter_response.cost > 0


@mark.parametrize("model_name", MODEL_NAMES)
async def test_async_assistant_first(
    mock_execute_async_responses: AsyncMock,  # pylint: disable=unused-argument,redefined-outer-name
    model_name: str,
) -> None:
    adapter = AdapterFactory.get_adapter_by_path(model_name)
    assert isinstance(adapter, GeminiSDKChatProviderAdapter)
    adapter_response = await adapter.execute_async(
        adapter.convert_to_input(SIMPLE_CONVERSATION_ASSISTANT_FIRST),
    )

    assert adapter_response == ASYNC_EXECUTE_EXPECTED_RESPONSES[model_name]


@mark.parametrize("model_name", MODEL_NAMES)
@mark.vcr
def test_sync_multiple_system(vcr, model_name: str) -> None:
    adapter = AdapterFactory.get_adapter_by_path(model_name)
    assert isinstance(adapter, GeminiSDKChatProviderAdapter)
    adapter_response = adapter.execute_sync(
        adapter.convert_to_input(SIMPLE_CONVERSATION_MULTIPLE_SYSTEM),
    )

    cassette_response = json.loads(vcr.responses[0]["body"]["string"])

    assert (
        adapter_response.response.content
        == cassette_response["candidates"][0]["content"]["parts"][0]["text"]
    )
    assert adapter_response.response.role == ConversationRole.assistant
    assert adapter_response.cost > 0


@mark.parametrize("model_name", MODEL_NAMES)
async def test_async_multiple_system(
    mock_execute_async_responses: AsyncMock,  # pylint: disable=unused-argument,redefined-outer-name
    model_name: str,
) -> None:
    adapter = AdapterFactory.get_adapter_by_path(model_name)
    assert isinstance(adapter, GeminiSDKChatProviderAdapter)
    adapter_response = await adapter.execute_async(
        adapter.convert_to_input(SIMPLE_CONVERSATION_MULTIPLE_SYSTEM),
    )

    assert adapter_response == ASYNC_EXECUTE_EXPECTED_RESPONSES[model_name]


@mark.parametrize("model_name", MODEL_NAMES)
@mark.vcr
def test_sync_empty_content(vcr, model_name: str) -> None:
    adapter = AdapterFactory.get_adapter_by_path(model_name)
    assert isinstance(adapter, GeminiSDKChatProviderAdapter)
    adapter_response = adapter.execute_sync(
        adapter.convert_to_input(SIMPLE_CONVERSATION_EMPTY_CONTENT),
    )

    cassette_response = json.loads(vcr.responses[0]["body"]["string"])

    assert (
        adapter_response.response.content
        == cassette_response["candidates"][0]["content"]["parts"][0]["text"]
    )
    assert adapter_response.response.role == ConversationRole.assistant
    assert adapter_response.cost > 0


@mark.parametrize("model_name", MODEL_NAMES)
async def test_async_empty_content(
    mock_execute_async_responses: AsyncMock,  # pylint: disable=unused-argument,redefined-outer-name
    model_name: str,
) -> None:
    adapter = AdapterFactory.get_adapter_by_path(model_name)
    assert isinstance(adapter, GeminiSDKChatProviderAdapter)
    adapter_response = await adapter.execute_async(
        adapter.convert_to_input(SIMPLE_CONVERSATION_EMPTY_CONTENT),
    )

    assert adapter_response == ASYNC_EXECUTE_EXPECTED_RESPONSES[model_name]


@mark.parametrize("model_name", MODEL_NAMES)
@mark.vcr
def test_sync_trailing_whitespace(vcr, model_name: str) -> None:
    adapter = AdapterFactory.get_adapter_by_path(model_name)
    assert isinstance(adapter, GeminiSDKChatProviderAdapter)
    adapter_response = adapter.execute_sync(
        adapter.convert_to_input(SIMPLE_CONVERSATION_TRAILING_WHITESPACE),
    )

    cassette_response = json.loads(vcr.responses[0]["body"]["string"])

    assert (
        adapter_response.response.content
        == cassette_response["candidates"][0]["content"]["parts"][0]["text"]
    )
    assert adapter_response.response.role == ConversationRole.assistant
    assert adapter_response.cost > 0


@mark.parametrize("model_name", MODEL_NAMES)
async def test_async_trailing_whitespace(
    mock_execute_async_responses: AsyncMock,  # pylint: disable=unused-argument,redefined-outer-name
    model_name: str,
) -> None:
    adapter = AdapterFactory.get_adapter_by_path(model_name)
    assert isinstance(adapter, GeminiSDKChatProviderAdapter)
    adapter_response = await adapter.execute_async(
        adapter.convert_to_input(SIMPLE_CONVERSATION_TRAILING_WHITESPACE),
    )

    assert adapter_response == ASYNC_EXECUTE_EXPECTED_RESPONSES[model_name]


@mark.parametrize("model_name", MODEL_NAMES)
@mark.vcr
def test_sync_system_assistant(vcr, model_name: str) -> None:
    adapter = AdapterFactory.get_adapter_by_path(model_name)
    assert isinstance(adapter, GeminiSDKChatProviderAdapter)
    adapter_response = adapter.execute_sync(
        adapter.convert_to_input(SIMPLE_CONVERSATION_ASSISTANT_SYSTEM),
    )

    cassette_response = json.loads(vcr.responses[0]["body"]["string"])

    assert (
        adapter_response.response.content
        == cassette_response["candidates"][0]["content"]["parts"][0]["text"]
    )
    assert adapter_response.response.role == ConversationRole.assistant
    assert adapter_response.cost > 0


@mark.parametrize("model_name", MODEL_NAMES)
async def test_async_system_assistant(
    mock_execute_async_responses: AsyncMock,  # pylint: disable=unused-argument,redefined-outer-name
    model_name: str,
) -> None:
    adapter = AdapterFactory.get_adapter_by_path(model_name)
    assert isinstance(adapter, GeminiSDKChatProviderAdapter)
    adapter_response = await adapter.execute_async(
        adapter.convert_to_input(SIMPLE_CONVERSATION_ASSISTANT_SYSTEM),
    )

    assert adapter_response == ASYNC_EXECUTE_EXPECTED_RESPONSES[model_name]


@mark.parametrize("model_name", MODEL_NAMES)
@mark.vcr
def test_sync_execute_json_content(vcr, model_name: str) -> None:
    adapter = AdapterFactory.get_adapter_by_path(model_name)
    assert isinstance(adapter, GeminiSDKChatProviderAdapter)
    adapter_response = adapter.execute_sync(
        adapter.convert_to_input(SIMPLE_CONVERSATION_JSON_CONTENT),
    )

    cassette_response = json.loads(vcr.responses[0]["body"]["string"])

    assert (
        adapter_response.response.content
        == cassette_response["candidates"][0]["content"]["parts"][0]["text"]
    )
    assert adapter_response.response.role == ConversationRole.assistant
    assert adapter_response.cost > 0


@mark.parametrize("model_name", MODEL_NAMES)
async def test_async_execute_json_content(
    mock_execute_async_responses: AsyncMock,  # pylint: disable=unused-argument,redefined-outer-name
    model_name: str,
) -> None:
    adapter = AdapterFactory.get_adapter_by_path(model_name)
    assert isinstance(adapter, GeminiSDKChatProviderAdapter)
    adapter_response = await adapter.execute_async(
        adapter.convert_to_input(SIMPLE_CONVERSATION_JSON_CONTENT),
    )

    assert adapter_response == ASYNC_EXECUTE_EXPECTED_RESPONSES[model_name]


@mark.parametrize("model_name", MODEL_NAMES)
@mark.vcr
def test_sync_execute_streaming(
    vcr, model_name: str  # pylint: disable=unused-argument
) -> None:
    adapter = AdapterFactory.get_adapter_by_path(model_name)
    assert isinstance(adapter, GeminiSDKChatProviderAdapter)
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


@mark.parametrize("model_name", MODEL_NAMES)
async def test_async_execute_streaming(
    mock_execute_async_responses: AsyncMock,  # pylint: disable=unused-argument,redefined-outer-name
    model_name: str,
) -> None:
    adapter = AdapterFactory.get_adapter_by_path(model_name)
    assert isinstance(adapter, GeminiSDKChatProviderAdapter)
    adapter_response = await adapter.execute_async(
        adapter.convert_to_input(SIMPLE_CONVERSATION_USER_ONLY),
        stream=True,
    )
    chunks = [
        json.loads(data_chunk[6:].strip())
        async for data_chunk in adapter_response.response
    ]

    expected_response = ASYNC_EXECUTE_EXPECTED_RESPONSES[model_name]
    assert chunks == [
        {
            "choices": [
                {
                    "delta": {
                        "role": expected_response.response.role,
                        "content": expected_response.response.content,
                    },
                },
            ],
        }
    ]
