import json
import re
from typing import Any, Dict, Pattern

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

PROVIDER_NAME = "anthropic"
BASE_URL = "https://api.anthropic.com"
API_KEY_NAME = "ANTHROPIC_API_KEY"
API_KEY_PATTERN = re.compile(r"^sk-ant-api\d{2}-([a-zA-Z0-9_-]+)$")


class AnthropicModel(Model):
    supports_streaming: bool = True
    supports_json_content: bool = True
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

    @staticmethod
    def get_base_sdk_url() -> str:
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
        choices = [
            {
                "message": {
                    "role": response.role,
                    "content": choice.text,
                },
                "finish_reason": FINISH_REASON_MAPPING.get(
                    response.stop_reason, response.stop_reason
                ),
            }
            for choice in response.content
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

    def get_params(
        self,
        llm_input: Conversation,
        **kwargs,
    ) -> Dict[str, Any]:
        params = super().get_params(llm_input, **kwargs)

        messages = params["messages"]
        system_prompt = ""

        # Convert json content to string
        for message in messages:
            if isinstance(message["content"], list):
                message["content"] = "\n".join(
                    [content["text"] for content in message["content"]]
                )

        if len(messages) > 0 and messages[0]["role"] == ConversationRole.system:
            system_prompt = messages[0]["content"]
            messages = messages[1:]

        # Extract system prompt if it's the first message
        if len(messages) > 0 and messages[0]["role"] == ConversationRole.system:
            system_prompt = messages[0]["content"]
            messages = messages[1:]

        # Change system prompt roles to assistant
        for message in messages:
            if message["role"] == ConversationRole.system:
                message["role"] = ConversationRole.assistant

        # Join messages from the same role
        processed_messages = []
        current_role = messages[0]["role"]
        current_content = messages[0]["content"]

        for message in messages[1:]:
            if message["role"] == current_role:
                current_content += "\n" + message["content"]
            else:
                # Otherwise, add the collected messages and reset for the next role
                processed_messages.append(
                    {"role": current_role, "content": current_content}
                )
                current_role = message["role"]
                current_content = message["content"]

        processed_messages.append({"role": current_role, "content": current_content})

        # Add empty user message if the first message is from the assistant
        if (
            len(processed_messages) > 0
            and processed_messages[0]["role"] == ConversationRole.assistant
        ):
            processed_messages.insert(
                0, {"role": ConversationRole.user, "content": "."}
            )

        # Remove trailing whitespace from the last assistant message
        if (
            len(processed_messages) > 0
            and processed_messages[-1]["role"] == ConversationRole.assistant
        ):
            processed_messages[-1]["content"] = processed_messages[-1][
                "content"
            ].rstrip()

        # If message is empty, use dot
        for message in processed_messages:
            if message["content"].strip() == "":
                message["content"] = "."

        return {
            **params,
            "messages": processed_messages,
            "system": system_prompt,
            "max_tokens": (
                kwargs.get("max_tokens")
                if kwargs.get("max_tokens")
                else self.get_model().completion_length
            ),
        }
