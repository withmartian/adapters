import json
import re
from typing import Any, Dict, List, Optional, Pattern

from anthropic import Anthropic, AsyncAnthropic

from adapters.abstract_adapters.api_key_adapter_mixin import ApiKeyAdapterMixin
from adapters.abstract_adapters.provider_adapter_mixin import ProviderAdapterMixin
from adapters.abstract_adapters.sdk_chat_adapter import SDKChatAdapter
from adapters.types import (
    Conversation,
    ConversationRole,
    Cost,
    Model,
    OpenAIChatAdapterResponse,
    Turn,
)
from adapters.utils.general_utils import process_image_url

PROVIDER_NAME = "anthropic"
BASE_URL = "https://api.anthropic.com"
API_KEY_NAME = "ANTHROPIC_API_KEY"
API_KEY_PATTERN = re.compile(r"^sk-ant-api\d{2}-([a-zA-Z0-9_-]+)$")


class AnthropicModel(Model):
    supports_streaming: bool = True
    supports_json_content: bool = True
    supports_empty_content: bool = False
    supports_first_assistant: bool = False
    supports_multiple_system: bool = False
    supports_repeating_roles: bool = False
    vendor_name: str = PROVIDER_NAME
    provider_name: str = PROVIDER_NAME


SUPPORTED_MODELS = [
    AnthropicModel(
        name="claude-1.0",
        cost=Cost(prompt=0.8e-6, completion=2.4e-6),
        context_length=9000,
        completion_length=2048,
    ),
    AnthropicModel(
        name="claude-1.1",
        cost=Cost(prompt=0.8e-6, completion=2.4e-6),
        context_length=9000,
        completion_length=2048,
    ),
    AnthropicModel(
        name="claude-1.2",
        cost=Cost(prompt=0.8e-6, completion=2.4e-6),
        context_length=9000,
        completion_length=2048,
    ),
    AnthropicModel(
        name="claude-1.3",
        cost=Cost(prompt=0.8e-6, completion=2.4e-6),
        context_length=9000,
        completion_length=2048,
    ),
    AnthropicModel(
        name="claude-instant-1.1",
        cost=Cost(prompt=0.8e-6, completion=2.4e-6),
        context_length=100000,
        completion_length=2048,
    ),
    AnthropicModel(
        name="claude-instant-1.2",
        cost=Cost(prompt=0.8e-6, completion=2.4e-6),
        context_length=100000,
        completion_length=4096,
    ),
    AnthropicModel(
        name="claude-2.0",
        cost=Cost(prompt=8.0e-6, completion=24.0e-6),
        context_length=100000,
        completion_length=4096,
    ),
    AnthropicModel(
        name="claude-2.1",
        cost=Cost(prompt=8.0e-6, completion=24.0e-6),
        context_length=200000,
        completion_length=4096,
    ),
    AnthropicModel(
        name="claude-3-sonnet-20240229",
        cost=Cost(prompt=3.0e-6, completion=15.0e-6),
        context_length=200000,
        completion_length=4096,
        supports_tools=True,
        supports_vision=True,
    ),
    AnthropicModel(
        name="claude-3-opus-20240229",
        cost=Cost(prompt=15.0e-6, completion=75.0e-6),
        context_length=200000,
        completion_length=4096,
        supports_tools=True,
        supports_vision=True,
    ),
    AnthropicModel(
        name="claude-3-haiku-20240307",
        cost=Cost(prompt=0.25e-6, completion=1.25e-6),
        context_length=200000,
        completion_length=4096,
        supports_tools=True,
        supports_vision=True,
    ),
    AnthropicModel(
        name="claude-3-5-sonnet-20240620",
        cost=Cost(prompt=3.0e-6, completion=15.0e-6),
        context_length=200000,
        completion_length=4096,
    ),
]

FINISH_REASON_MAPPING = {
    "end_turn": "stop",
    "max_tokens": "length",
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

    @staticmethod
    def get_api_key_pattern() -> Pattern:
        return API_KEY_PATTERN

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

    def get_async_client(self):
        return self._async_client.messages.create

    def get_sync_client(self):
        return self._sync_client.messages.create

    def adjust_temperature(self, temperature: float) -> float:
        return temperature / 2

    def set_api_key(self, api_key: str) -> None:
        super().set_api_key(api_key)

        self._sync_client.api_key = api_key
        self._async_client.api_key = api_key

    def extract_response(
        self, request: Any, response: Any
    ) -> OpenAIChatAdapterResponse:
        tool_call_name = None
        arguments = None
        if response.content[0].type == "tool_use":
            response.content[0].text = ""
            tool_call_name = response.content[0].name
            arguments = response.content[0].input

        choices = [
            {
                "message": {
                    "role": response.role,
                    "content": response.content[0].text,
                    "tool_calls": (
                        [
                            {
                                "function": {
                                    "name": tool_call_name,
                                    "arguments": arguments,
                                }
                            }
                        ]
                        if tool_call_name
                        else []
                    ),
                },
                "finish_reason": FINISH_REASON_MAPPING.get(
                    response.stop_reason, response.stop_reason
                ),
            }
        ]
        prompt_tokens = self._sync_client.count_tokens(
            request.convert_to_anthropic_prompt()
        )
        completion_tokens = self._sync_client.count_tokens(response.content[0].text)
        cost = (
            self.get_model().cost.prompt * prompt_tokens
            + self.get_model().cost.completion * completion_tokens
            + self.get_model().cost.request
        )

        return OpenAIChatAdapterResponse(
            response=Turn(
                role=ConversationRole.assistant,
                content=choices[0]["message"]["content"],  # type: ignore
            ),  # TODO: Refactor response
            choices=choices,
            cost=cost,
            token_counts=Cost(
                prompt=prompt_tokens,
                completion=completion_tokens,
            ),
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
