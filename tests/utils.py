import json
from typing import Any, Iterable

import brotli

from adapters.abstract_adapters.base_adapter import BaseAdapter
from adapters.abstract_adapters.openai_sdk_chat_adapter import OpenAISDKChatAdapter
from adapters.adapter_factory import AdapterFactory
from adapters.provider_adapters.ai21_sdk_chat_provider_adapter import AI21Model
from adapters.provider_adapters.anthropic_sdk_chat_provider_adapter import (
    AnthropicModel,
    AnthropicSDKChatProviderAdapter,
)
from adapters.provider_adapters.cerebras_sdk_chat_provider_adapter import CerebrasModel
from adapters.provider_adapters.cohere_sdk_chat_provider_adapter import (
    CohereModel,
    CohereSDKChatProviderAdapter,
)
from adapters.provider_adapters.deepinfra_sdk_chat_provider_adapter import (
    DeepInfraModel,
)
from adapters.provider_adapters.fireworks_sdk_chat_provider_adapter import (
    FireworksModel,
)
from adapters.provider_adapters.gemini_sdk_chat_provider_adapter import (
    GeminiSDKChatProviderAdapter,
)
from adapters.provider_adapters.moescape_sdk_chat_provider_adapter import MoescapeModel
from adapters.provider_adapters.moonshot_sdk_chat_provider_adapter import MoonshotModel
from adapters.provider_adapters.openai_sdk_chat_provider_adapter import OpenAIModel
from vcr import VCR
from openai.types.chat import ChatCompletionMessageParam

from adapters.provider_adapters.openrouter_sdk_chat_provider_adapter import (
    OpenRouterModel,
)
from adapters.provider_adapters.together_sdk_chat_provider_adapter import TogetherModel


class AdapterTestFactory:
    model_path: str

    def __init__(self, model_path: str):
        self.model_path = model_path

    def __call__(self) -> BaseAdapter:
        adapter = AdapterFactory.get_adapter_by_path(self.model_path)
        if adapter is None:
            raise ValueError(f"No adapter found for path: {self.model_path}")
        return adapter

    def __str__(self) -> str:
        return self.model_path


TEST_MODELS = (
    OpenAIModel,
    AnthropicModel,
    TogetherModel,
    AI21Model,
    CerebrasModel,
    CohereModel,
    FireworksModel,
    MoescapeModel,
    # GeminiModel,
    # PerplexityModel,
    OpenRouterModel,
    MoonshotModel,
    # LeptonModel,
    DeepInfraModel,
)

ADAPTER_CHAT_TEST_FACTORIES = [
    AdapterTestFactory(model.get_path())
    for model in AdapterFactory.get_supported_models()
    if isinstance(model, TEST_MODELS) and model.supports_chat
]

ADAPTER_COMPLETION_TEST_FACTORIES = [
    AdapterTestFactory(model.get_path())
    for model in AdapterFactory.get_supported_models()
    if isinstance(model, TEST_MODELS) and model.supports_completion
]


SIMPLE_CONVERSATION_USER_ONLY: Iterable[ChatCompletionMessageParam] = [
    {
        "role": "user",
        "content": "Hi",
    }
]


SIMPLE_CONVERSATION_JSON_OUTPUT: Iterable[ChatCompletionMessageParam] = [
    {
        "role": "user",
        "content": "Hi, output simple json",
    }
]


SIMPLE_CONVERSATION_JSON_CONTENT: Iterable[ChatCompletionMessageParam] = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Hi",
            },
            {
                "type": "text",
                "text": "Hi",
            },
        ],
    }
]


SIMPLE_FUNCTION_CALL_USER_ONLY: Iterable[ChatCompletionMessageParam] = [
    {
        "role": "user",
        "content": "Generate random number with tools specified, range 1-10",
    }
]

SIMPLE_GENERATE_TOOLS = [
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
]


def get_response_content_from_vcr(vcr: VCR, adapter: BaseAdapter) -> Any:
    response = vcr.responses[-1]["body"]["string"]

    try:
        response = brotli.decompress(response)
    except Exception as _:  # pylint: disable=W0718
        print("Failed to decompress response")

    response = json.loads(response)

    if isinstance(adapter, OpenAISDKChatAdapter):
        return response["choices"][0]["message"]["content"]
    elif isinstance(adapter, AnthropicSDKChatProviderAdapter):
        return response["content"][0]["text"]
    elif isinstance(adapter, CohereSDKChatProviderAdapter):
        return (
            response["message"]["content"][0]["text"]
            if response.get("message") and response["message"].get("content")
            else ""
        )
    elif isinstance(adapter, GeminiSDKChatProviderAdapter):
        return response["candidates"][0]["content"]["parts"][0]["text"]
    else:
        raise ValueError("Unknown adapter")


def get_response_choices_from_vcr(vcr: VCR, adapter: BaseAdapter) -> Any:
    response = vcr.responses[-1]["body"]["string"]

    try:
        response = brotli.decompress(response)
    except Exception as _:  # pylint: disable=W0718
        print("Failed to decompress response")

    response = json.loads(response)

    if isinstance(adapter, OpenAISDKChatAdapter):
        return response["choices"]
    elif isinstance(adapter, AnthropicSDKChatProviderAdapter):
        anthropic_choices: list[Any] = []

        for content in response["content"]:
            if content["type"] == "tool_use":
                anthropic_choices.append(
                    {
                        "message": {
                            "role": response["role"],
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": content["name"],
                                        "arguments": json.dumps(content["input"]),
                                    },
                                }
                            ],
                        }
                    }
                )
            elif content["type"] == "text":
                anthropic_choices.append(
                    {
                        "message": {
                            "role": response["role"],
                            "content": content["text"],
                        }
                    }
                )
            else:
                raise ValueError(f"Unknown content type: {content['type']}")
        return anthropic_choices
    elif isinstance(adapter, CohereSDKChatProviderAdapter):
        # Parse the Cohere SDK response
        cohere_choices: list[Any] = []

        # Assuming the structure of the Cohere response can vary
        message = response.get("message")
        role = message.get("role", "assistant")
        tool_calls = message.get("tool_calls")
        content_list = message.get("content")

        # Check if there are tool calls and handle them
        if tool_calls:
            for tool_call in tool_calls:
                cohere_choices.append(
                    {
                        "message": {
                            "role": role,
                            "tool_calls": [
                                {
                                    "id": tool_call["id"],
                                    "type": tool_call["type"],
                                    "function": {
                                        "name": tool_call["function"]["name"],
                                        "arguments": tool_call["function"]["arguments"],
                                    },
                                }
                            ],
                        }
                    }
                )
        # If no tool calls, check for and handle content as a list of text items
        elif content_list:
            for content in content_list:
                if content.get("type") == "text":
                    cohere_choices.append(
                        {"message": {"role": role, "content": content["text"]}}
                    )

        return cohere_choices

    # elif isinstance(adapter, GeminiSDKChatProviderAdapter):
    # return response["candidates"][0]["content"]["parts"][0]["text"]
    else:
        raise ValueError("Unknown adapter")
