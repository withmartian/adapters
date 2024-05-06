import json
from typing import List

import pytest

from adapters import AdapterFactory
from adapters.provider_adapters.anyscale_sdk_chat_provider_adapter import (
    AnyscaleSDKChatProviderAdapter,
)
from adapters.provider_adapters.deepinfra_sdk_chat_provider_adapter import (
    DeepInfraSDKChatProviderAdapter,
)
from adapters.provider_adapters.fireworks_sdk_chat_provider_adapter import (
    FireworksSDKChatProviderAdapter,
)
from adapters.provider_adapters.groq_sdk_chat_provider_adapter import (
    GroqSDKChatProviderAdapter,
)
from adapters.provider_adapters.lepton_sdk_chat_provider_adapter import (
    LeptonSDKChatProviderAdapter,
)
from adapters.provider_adapters.moonshot_sdk_chat_provider_adapter import (
    MoonshotSDKChatProviderAdapter,
)
from adapters.provider_adapters.octoai_sdk_chat_provider_adapter import (
    OctoaiSDKChatProviderAdapter,
)
from adapters.provider_adapters.openai_sdk_chat_provider_adapter import (
    OpenAISDKChatProviderAdapter,
)
from adapters.provider_adapters.openrouter_sdk_chat_provider_adapter import (
    OpenRouterSDKChatProviderAdapter,
)
from adapters.provider_adapters.perplexity_sdk_chat_provider_adapter import (
    PerplexitySDKChatProviderAdapter,
)
from adapters.provider_adapters.together_sdk_chat_provider_adapter import (
    TogetherSDKChatProviderAdapter,
)
from adapters.types import AdapterException, ConversationRole, Model
from tests.utils import (
    SIMPLE_CONVERSATION_JSON,
    SIMPLE_CONVERSATION_JSON_CONTENT,
    SIMPLE_CONVERSATION_USER_ONLY,
    SIMPLE_CONVERSATION_VISION,
    SIMPLE_FUNCTION_CALL_USER_ONLY,
    get_choices_from_vcr,
)

MODELS: List[Model] = [
    *OpenAISDKChatProviderAdapter.get_supported_models(),
    *AnyscaleSDKChatProviderAdapter.get_supported_models(),
    *DeepInfraSDKChatProviderAdapter.get_supported_models(),
    *OpenRouterSDKChatProviderAdapter.get_supported_models(),
    *TogetherSDKChatProviderAdapter.get_supported_models(),
    *PerplexitySDKChatProviderAdapter.get_supported_models(),
    *GroqSDKChatProviderAdapter.get_supported_models(),
    *LeptonSDKChatProviderAdapter.get_supported_models(),
    *MoonshotSDKChatProviderAdapter.get_supported_models(),
    *FireworksSDKChatProviderAdapter.get_supported_models(),
    *OctoaiSDKChatProviderAdapter.get_supported_models(),
]

MODEL_NAMES = [
    f"{model.provider_name}/{model.vendor_name}/{model.name}" for model in MODELS
]

OPTIONS_TO_GENERATE = 2
MAX_TOKENS = 5

# Change response to choices


@pytest.mark.parametrize("model_name", MODEL_NAMES)
@pytest.mark.vcr
def test_sync_execute(vcr, model_name):
    adapter = AdapterFactory.get_adapter(model_name)
    adapter_response = adapter.execute_sync(
        adapter.convert_to_input(SIMPLE_CONVERSATION_USER_ONLY)
    )

    choices = get_choices_from_vcr(vcr)

    assert adapter_response.response.content == choices[0]["message"]["content"]
    assert adapter_response.response.role == ConversationRole.assistant
    assert adapter_response.cost > 0


@pytest.mark.parametrize("model_name", MODEL_NAMES)
@pytest.mark.vcr
async def test_async_execute(vcr, model_name):
    adapter = AdapterFactory.get_adapter(model_name)
    adapter_response = await adapter.execute_async(
        adapter.convert_to_input(SIMPLE_CONVERSATION_USER_ONLY)
    )

    choices = get_choices_from_vcr(vcr)

    assert adapter_response.response.content == choices[0]["message"]["content"]
    assert adapter_response.response.role == ConversationRole.assistant
    assert adapter_response.cost > 0


# @pytest.mark.parametrize("model_name", MODEL_NAMES)
# @pytest.mark.vcr
# def test_sync_execute_finish_reason(vcr, model_name):
#     MAX_TOKENS_FINISH_LENGTH = 1
#     MAX_TOKENS_FINISH_STOP = 100

#     adapter = AdapterFactory.get_adapter(model_name)
#     adapter.execute_sync(
#         adapter.convert_to_input(SIMPLE_CONVERSATION_USER_ONLY),
#         max_tokens=MAX_TOKENS_FINISH_LENGTH,
#     )

#     choices_finish_length = get_choices_from_vcr(vcr)

#     assert choices_finish_length[0]["finish_reason"] == FinishReason.length

#     adapter.execute_sync(
#         adapter.convert_to_input(SIMPLE_CONVERSATION_USER_ONLY),
#         max_tokens=MAX_TOKENS_FINISH_STOP,
#     )

#     choices_finish_stop = get_choices_from_vcr(vcr)

#     assert choices_finish_stop[0]["finish_reason"] == FinishReason.stop


@pytest.mark.parametrize("model_name", MODEL_NAMES)
@pytest.mark.vcr
def test_sync_execute_n(vcr, model_name):
    adapter = AdapterFactory.get_adapter(model_name)

    if adapter.supports_n() is False:
        return

    adapter_response = adapter.execute_sync(
        adapter.convert_to_input(SIMPLE_CONVERSATION_USER_ONLY), n=OPTIONS_TO_GENERATE
    )

    choices = get_choices_from_vcr(vcr)

    assert adapter_response.response.content == choices[0]["message"]["content"]
    assert adapter_response.response.role == ConversationRole.assistant
    assert adapter_response.cost > 0
    assert len(choices) == OPTIONS_TO_GENERATE


@pytest.mark.parametrize("model_name", MODEL_NAMES)
@pytest.mark.vcr
async def test_async_execute_n(vcr, model_name):
    adapter = AdapterFactory.get_adapter(model_name)

    if adapter.supports_n() is False:
        return

    adapter_response = await adapter.execute_async(
        adapter.convert_to_input(SIMPLE_CONVERSATION_USER_ONLY), n=OPTIONS_TO_GENERATE
    )

    choices = get_choices_from_vcr(vcr)

    assert adapter_response.response.content == choices[0]["message"]["content"]
    assert adapter_response.response.role == ConversationRole.assistant
    assert adapter_response.cost > 0
    assert len(choices) == OPTIONS_TO_GENERATE


@pytest.mark.parametrize("model_name", MODEL_NAMES)
@pytest.mark.vcr
def test_sync_execute_streaming(model_name):
    adapter = AdapterFactory.get_adapter(model_name)

    if adapter.supports_streaming() is False:
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


@pytest.mark.parametrize("model_name", MODEL_NAMES)
@pytest.mark.vcr
async def test_async_execute_streaming(model_name):
    adapter = AdapterFactory.get_adapter(model_name)

    if adapter.supports_streaming() is False:
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


@pytest.mark.parametrize("model_name", MODEL_NAMES)
@pytest.mark.vcr
def test_sync_execute_function_calls(vcr, model_name):
    adapter = AdapterFactory.get_adapter(model_name)

    if adapter.supports_functions() is False:
        return

    adapter_response = adapter.execute_sync(
        adapter.convert_to_input(SIMPLE_FUNCTION_CALL_USER_ONLY),
        function_call={"name": "generate"},
        functions=[{"description": "Generate random number", "name": "generate"}],
    )
    choices = get_choices_from_vcr(vcr)

    assert (
        adapter_response.choices[0].message.function_call.name
        == choices[0]["message"]["function_call"]["name"]
    )
    assert (
        adapter_response.choices[0].message.function_call.arguments
        == choices[0]["message"]["function_call"]["arguments"]
    )
    assert adapter_response.choices[0].message.role == ConversationRole.assistant
    assert adapter_response.cost > 0


@pytest.mark.parametrize("model_name", MODEL_NAMES)
@pytest.mark.vcr
async def test_async_execute_function_calls(vcr, model_name):
    adapter = AdapterFactory.get_adapter(model_name)

    if adapter.supports_functions() is False:
        return

    adapter_response = await adapter.execute_async(
        adapter.convert_to_input(SIMPLE_FUNCTION_CALL_USER_ONLY),
        function_call={"name": "generate"},
        functions=[{"description": "Generate random number", "name": "generate"}],
    )
    choices = get_choices_from_vcr(vcr)

    assert (
        adapter_response.choices[0].message.function_call.name
        == choices[0]["message"]["function_call"]["name"]
    )
    assert (
        adapter_response.choices[0].message.function_call.arguments
        == choices[0]["message"]["function_call"]["arguments"]
    )
    assert adapter_response.choices[0].message.role == ConversationRole.assistant
    assert adapter_response.cost > 0


@pytest.mark.parametrize("model_name", MODEL_NAMES)
@pytest.mark.vcr
def test_sync_execute_tools(vcr, model_name):
    adapter = AdapterFactory.get_adapter(model_name)

    if adapter.supports_tools() is False:
        return

    adapter_response = adapter.execute_sync(
        adapter.convert_to_input(SIMPLE_FUNCTION_CALL_USER_ONLY),
        tool_choice={"type": "function", "function": {"name": "generate"}},
        tools=[
            {
                "type": "function",
                "function": {
                    "description": "Generate random number",
                    "name": "generate",
                },
            }
        ],
    )
    choices = get_choices_from_vcr(vcr)

    assert (
        adapter_response.choices[0].message.tool_calls[0].function.name
        == choices[0]["message"]["tool_calls"][0]["function"]["name"]
    )
    assert (
        adapter_response.choices[0].message.tool_calls[0].function.arguments
        == choices[0]["message"]["tool_calls"][0]["function"]["arguments"]
    )
    assert adapter_response.choices[0].message.role == ConversationRole.assistant
    assert adapter_response.cost > 0


@pytest.mark.parametrize("model_name", MODEL_NAMES)
@pytest.mark.vcr
async def test_async_execute_tools(vcr, model_name):
    adapter = AdapterFactory.get_adapter(model_name)

    if adapter.supports_tools() is False:
        return

    adapter_response = await adapter.execute_async(
        adapter.convert_to_input(SIMPLE_FUNCTION_CALL_USER_ONLY),
        tool_choice={"type": "function", "function": {"name": "generate"}},
        tools=[
            {
                "type": "function",
                "function": {
                    "description": "Generate random number",
                    "name": "generate",
                },
            }
        ],
    )
    choices = get_choices_from_vcr(vcr)

    assert (
        adapter_response.choices[0].message.tool_calls[0].function.name
        == choices[0]["message"]["tool_calls"][0]["function"]["name"]
    )
    assert (
        adapter_response.choices[0].message.tool_calls[0].function.arguments
        == choices[0]["message"]["tool_calls"][0]["function"]["arguments"]
    )
    assert adapter_response.choices[0].message.role == ConversationRole.assistant
    assert adapter_response.cost > 0


@pytest.mark.parametrize("model_name", MODEL_NAMES)
@pytest.mark.vcr
def test_sync_execute_vision(vcr, model_name):
    adapter = AdapterFactory.get_adapter(model_name)

    if adapter.supports_vision() is False:
        return

    adapter_response = adapter.execute_sync(
        adapter.convert_to_input(SIMPLE_CONVERSATION_VISION)
    )
    choices = get_choices_from_vcr(vcr)

    assert adapter_response.response.content == choices[0]["message"]["content"]
    assert adapter_response.response.role == ConversationRole.assistant
    assert adapter_response.cost > 0


@pytest.mark.parametrize("model_name", MODEL_NAMES)
@pytest.mark.vcr
async def test_async_execute_vision(vcr, model_name):
    adapter = AdapterFactory.get_adapter(model_name)

    if adapter.supports_vision() is False:
        return

    adapter_response = await adapter.execute_async(
        adapter.convert_to_input(SIMPLE_CONVERSATION_VISION)
    )
    choices = get_choices_from_vcr(vcr)

    assert adapter_response.response.content == choices[0]["message"]["content"]
    assert adapter_response.response.role == ConversationRole.assistant
    assert adapter_response.cost > 0


@pytest.mark.parametrize("model_name", MODEL_NAMES)
@pytest.mark.vcr
def test_sync_execute_json_output(vcr, model_name):
    adapter = AdapterFactory.get_adapter(model_name)

    if adapter.supports_json_output() is False:
        return

    adapter_response = adapter.execute_sync(
        adapter.convert_to_input(SIMPLE_CONVERSATION_JSON),
        response_format={"type": "json_object"},
    )
    choices = get_choices_from_vcr(vcr)

    assert json.loads(adapter_response.response.content)
    assert adapter_response.response.content == choices[0]["message"]["content"]
    assert adapter_response.response.role == ConversationRole.assistant
    assert adapter_response.cost > 0


@pytest.mark.parametrize("model_name", MODEL_NAMES)
@pytest.mark.vcr
async def test_async_execute_json_output(vcr, model_name):
    adapter = AdapterFactory.get_adapter(model_name)

    if adapter.supports_json_output() is False:
        return

    adapter_response = await adapter.execute_async(
        adapter.convert_to_input(SIMPLE_CONVERSATION_JSON),
        response_format={"type": "json_object"},
    )
    choices = get_choices_from_vcr(vcr)

    assert json.loads(adapter_response.response.content)
    assert adapter_response.response.content == choices[0]["message"]["content"]
    assert adapter_response.response.role == ConversationRole.assistant
    assert adapter_response.cost > 0


@pytest.mark.parametrize("model_name", MODEL_NAMES)
@pytest.mark.vcr
def test_sync_execute_json_content(vcr, model_name):
    adapter = AdapterFactory.get_adapter(model_name)

    if adapter.supports_json_content() is False:
        return

    adapter_response = adapter.execute_sync(
        adapter.convert_to_input(SIMPLE_CONVERSATION_USER_ONLY),
    )
    choices = get_choices_from_vcr(vcr)

    assert adapter_response.response.content == choices[0]["message"]["content"]
    assert adapter_response.response.role == ConversationRole.assistant
    assert adapter_response.cost > 0


@pytest.mark.parametrize("model_name", MODEL_NAMES)
@pytest.mark.vcr
async def test_async_execute_json_content(vcr, model_name):
    adapter = AdapterFactory.get_adapter(model_name)

    if adapter.supports_json_content() is False:
        return

    adapter_response = await adapter.execute_async(
        adapter.convert_to_input(SIMPLE_CONVERSATION_JSON_CONTENT),
    )
    choices = get_choices_from_vcr(vcr)

    assert adapter_response.response.content == choices[0]["message"]["content"]
    assert adapter_response.response.role == ConversationRole.assistant
    assert adapter_response.cost > 0


@pytest.mark.parametrize("model_name", MODEL_NAMES)
@pytest.mark.vcr
async def test_async_stream_exceptions_closes_connection_ok(model_name):
    adapter = AdapterFactory.get_adapter(model_name)

    if adapter.supports_streaming is False:
        return

    # Monkey patch the extract_stream_response method to raise an exception
    def extract_stream_response(self, request, response):
        raise ValueError("Simulated exception")

    adapter.extract_stream_response = (
        extract_stream_response.__get__(  # pylint: disable=no-value-for-parameter
            adapter, adapter.__class__.__name__
        )
    )

    adapter_response = await adapter.execute_async(
        adapter.convert_to_input(SIMPLE_CONVERSATION_USER_ONLY),
        stream=True,
    )

    with pytest.raises(AdapterException, match="Error in streaming response"):
        async for _ in adapter_response.response:
            pass
