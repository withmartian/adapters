from enum import Enum
import json
import re
import time
from typing import Any, Dict, List, Literal, Optional, Pattern

from anthropic import Anthropic, AsyncAnthropic
from anthropic.types import Message
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)

from adapters.abstract_adapters.api_key_adapter_mixin import ApiKeyAdapterMixin
from adapters.abstract_adapters.provider_adapter_mixin import ProviderAdapterMixin
from adapters.abstract_adapters.sdk_chat_adapter import SDKChatAdapter
from adapters.types import (
    AdapterChatCompletion,
    Conversation,
    ConversationRole,
    Cost,
    FinishReason,
    Model,
    ModelPredicates,
)
from adapters.utils.general_utils import process_image_url

PROVIDER_NAME = "anthropic"
BASE_URL = "https://api.anthropic.com"
API_KEY_NAME = "ANTHROPIC_API_KEY"
BASE_PREDICATES = ModelPredicates(gdpr_compliant=True)


class AnthropicModel(Model):
    vendor_name: str = PROVIDER_NAME
    provider_name: str = PROVIDER_NAME
    predicates: ModelPredicates = BASE_PREDICATES

    supports_tool_choice_required: bool = True
    supports_last_assistant: bool = True
    supports_streaming: bool = True
    supports_json_content: bool = True
    supports_tools: bool = True
    supports_vision: bool = True


SUPPORTED_MODELS = [
    AnthropicModel(
        name="claude-3-sonnet-20240229",
        cost=Cost(prompt=3.0e-6, completion=15.0e-6),
        context_length=200000,
        completion_length=4096,
    ),
    AnthropicModel(
        name="claude-3-opus-20240229",
        cost=Cost(prompt=15.0e-6, completion=75.0e-6),
        context_length=200000,
        completion_length=4096,
    ),
    AnthropicModel(
        name="claude-3-haiku-20240307",
        cost=Cost(prompt=0.25e-6, completion=1.25e-6),
        context_length=200000,
        completion_length=4096,
    ),
    AnthropicModel(
        name="claude-3-5-sonnet-20240620",
        cost=Cost(prompt=3.0e-6, completion=15.0e-6),
        context_length=200000,
        completion_length=4096,
    ),
]


class AnthropicFinishReason(str, Enum):
    end_turn = "end_turn"
    max_tokens = "max_tokens"
    stop_sequence = "stop_sequence"
    tool_use = "tool_use"


FINISH_REASON_MAPPING = {
    AnthropicFinishReason.end_turn: FinishReason.stop,
    AnthropicFinishReason.max_tokens: FinishReason.length,
    AnthropicFinishReason.stop_sequence: FinishReason.stop,
    AnthropicFinishReason.tool_use: FinishReason.tool_calls,
}


class AnthropicSDKChatProviderAdapter(
    ProviderAdapterMixin,
    ApiKeyAdapterMixin,
    SDKChatAdapter,
):
    @staticmethod
    def get_supported_models():
        return SUPPORTED_MODELS

    @staticmethod
    def get_provider_name() -> str:
        return PROVIDER_NAME

    def get_base_sdk_url(self) -> str:
        return BASE_URL

    @staticmethod
    def get_api_key_name() -> str:
        return API_KEY_NAME

    _sync_client: Anthropic
    _async_client: AsyncAnthropic

    def __init__(
        self,
    ):
        super().__init__()
        self._sync_client = Anthropic(
            api_key=self.get_api_key(),
        )
        self._async_client = AsyncAnthropic(
            api_key=self.get_api_key(),
        )

    def get_sync_client(self):
        return self._sync_client.messages.create

    def get_async_client(self):
        return self._async_client.messages.create

    def adjust_temperature(self, temperature: float) -> float:
        return temperature / 2

    def set_api_key(self, api_key: str) -> None:
        super().set_api_key(api_key)

        self._sync_client.api_key = api_key
        self._async_client.api_key = api_key

    def extract_response(
        self, request: Conversation, response: Message
    ) -> AdapterChatCompletion:
        finish_reason = FINISH_REASON_MAPPING.get(
            response.stop_reason, FinishReason.stop
        )

        choices: list[Choice] = []
        for content in response.content:
            if content.type == "text":
                choices.append(
                    Choice(
                        index=len(choices),
                        finish_reason=finish_reason,
                        message=ChatCompletionMessage(
                            role=ConversationRole.assistant,
                            content=content.text,
                        ),
                    )
                )
            elif content.type == "tool_use":
                choices.append(
                    Choice(
                        index=len(choices),
                        finish_reason=finish_reason,
                        message=ChatCompletionMessage(
                            role=ConversationRole.assistant,
                            tool_calls=[
                                ChatCompletionMessageToolCall(
                                    id=content.id,
                                    type="function",
                                    function=Function(
                                        name=content.name,
                                        arguments=json.dumps(content.input),
                                    ),
                                )
                            ],
                        ),
                    )
                )

        usage = CompletionUsage(
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
        )

        cost = (
            self.get_model().cost.prompt * usage.prompt_tokens
            + self.get_model().cost.completion * usage.completion_tokens
            + self.get_model().cost.request
        )

        return AdapterChatCompletion(
            id=response.id,
            created=int(time.time()),
            model=self.get_model().name,
            object="chat.completion",
            cost=cost,
            usage=usage,
            choices=choices,
        )

    # TODO: match openai format 1:1
    def extract_stream_response(self, request, response):
        content = getattr(getattr(response, "delta", None), "text", "")

        if getattr(response, "type", None) == "message_stop":
            content = None

        chunk = json.dumps(
            {
                "choices": [
                    {
                        "delta": {
                            "role": ConversationRole.assistant,
                            "content": content,
                        },
                    }
                ]
            }
        )

        return f"data: {chunk}\n\n"

    def get_params(self, llm_input: Conversation, **kwargs) -> Dict[str, Any]:
        params = super().get_params(llm_input, **kwargs)
        messages = params["messages"]
        system_prompt = ""
        # Extract system prompt if it's the first message
        if len(messages) > 0 and messages[0]["role"] == ConversationRole.system:
            system_prompt = messages[0]["content"]
            messages = messages[1:]

        # Remove trailing whitespace from the last assistant message
        if len(messages) > 0 and messages[-1]["role"] == ConversationRole.assistant:
            messages[-1]["content"] = messages[-1]["content"].rstrip()

        anthropic_tools: Optional[List[Dict[str, Any]]] = kwargs.get("tools")
        anthropic_tools_choice = kwargs.get("tool_choice")
        if anthropic_tools_choice and not isinstance(anthropic_tools_choice, str):
            anthropic_tools_choice["name"] = anthropic_tools_choice["function"]["name"]
            anthropic_tools_choice["type"] = "tool"
            del anthropic_tools_choice["function"]
        elif anthropic_tools_choice == "required":
            anthropic_tools_choice = {"type": "any"}
        elif anthropic_tools_choice == "auto":
            anthropic_tools_choice = {"type": "auto"}
        else:
            anthropic_tools_choice = {"type": "auto"}
            anthropic_tools = []

        if anthropic_tools:
            for tool in anthropic_tools:
                tool["name"] = tool["function"]["name"]
                tool["description"] = tool["function"]["description"]
                tool["input_schema"] = {
                    "type": tool["function"]["parameters"]["type"],
                    "properties": tool["function"]["parameters"]["properties"],
                    "required": tool["function"]["parameters"]["required"],
                }
                del tool["function"]
                del tool["type"]
        else:
            anthropic_tools_choice = None

        # Include base64-encoded images in the request
        for message in messages:
            if message["role"] == ConversationRole.user:
                new_content = []

                if isinstance(message["content"], list):
                    for content in message["content"]:
                        if content["type"] == "text":
                            new_content.append(content)
                        elif content["type"] == "image_url":
                            new_content.append(
                                process_image_url(content["image_url"]["url"])
                            )
                else:
                    new_content = [{"type": "text", "text": message["content"]}]

                message["content"] = new_content

        return {
            **params,
            "messages": messages,
            "system": system_prompt,
            "max_tokens": kwargs.get("max_tokens", self.get_model().completion_length),
            "tool_choice": anthropic_tools_choice,
            "tools": anthropic_tools,
        }
