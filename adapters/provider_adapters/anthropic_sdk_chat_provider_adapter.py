from enum import Enum
import json
import time
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Union,
)

from anthropic import Anthropic, AsyncAnthropic
from anthropic.types import (
    Message,
    RawContentBlockDeltaEvent,
    RawMessageDeltaEvent,
    RawMessageStartEvent,
    RawMessageStreamEvent,
)
from anthropic.types.message_create_params import (
    Metadata,
    ToolChoice,
    ToolChoiceToolChoiceAny,
    ToolChoiceToolChoiceAuto,
    ToolChoiceToolChoiceTool,
)
from anthropic.types.message_param import MessageParam
from anthropic.types.raw_content_block_delta_event import (
    TextDelta,
)
from anthropic.types.text_block_param import TextBlockParam
from anthropic.types.tool_param import ToolParam
from httpx import AsyncClient, Client, Limits, Timeout
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import Choice as ChoiceChunk, ChoiceDelta
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)
from pydantic import BaseModel

from adapters.abstract_adapters.sdk_chat_adapter import SDKChatAdapter
from adapters.constants import (
    HTTP_CONNECT_TIMEOUT,
    HTTP_TIMEOUT,
    MAX_CONNECTIONS_PER_PROCESS,
    MAX_KEEPALIVE_CONNECTIONS_PER_PROCESS,
)
from adapters.general_utils import process_image_url_anthropic
from adapters.types import (
    AdapterChatCompletion,
    AdapterChatCompletionChunk,
    AdapterFinishReason,
    Conversation,
    ConversationRole,
    Cost,
    Model,
    ModelProperties,
    Turn,
)

PROVIDER_NAME = "anthropic"
BASE_URL = "https://api.anthropic.com"
API_KEY_NAME = "ANTHROPIC_API_KEY"
BASE_PROPERTIES = ModelProperties(gdpr_compliant=True)


class AnthropicModel(Model):
    vendor_name: str = PROVIDER_NAME
    provider_name: str = PROVIDER_NAME
    properties: ModelProperties = BASE_PROPERTIES

    supports_tool_choice_required: bool = True
    supports_last_assistant: bool = True
    supports_streaming: bool = True
    supports_json_content: bool = False
    supports_tools: bool = True
    supports_vision: bool = True
    supports_temperature: bool = True


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
    AnthropicModel(
        name="claude-3-5-sonnet-20241022",
        cost=Cost(prompt=3.0e-6, completion=15.0e-6),
        context_length=200000,
        completion_length=4096,
    ),
    AnthropicModel(
        name="claude-3-5-sonnet-latest",
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


FINISH_REASON_MAPPING: Dict[AnthropicFinishReason, AdapterFinishReason] = {
    AnthropicFinishReason.end_turn: AdapterFinishReason.stop,
    AnthropicFinishReason.max_tokens: AdapterFinishReason.length,
    AnthropicFinishReason.stop_sequence: AdapterFinishReason.stop,
    AnthropicFinishReason.tool_use: AdapterFinishReason.tool_calls,
}


class AnthropicCreate(BaseModel):
    max_tokens: int
    messages: Iterable[MessageParam]
    metadata: Optional[Metadata] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[Literal[False] | Literal[True]] = None
    system: Optional[Union[str, Iterable[TextBlockParam]]] = None
    temperature: Optional[float] = None
    tool_choice: Optional[ToolChoice] = None
    tools: Optional[Iterable[ToolParam]] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None


class AnthropicSDKChatProviderAdapter(SDKChatAdapter[Anthropic, AsyncAnthropic]):
    @staticmethod
    def get_supported_models():
        return SUPPORTED_MODELS

    @staticmethod
    def get_provider_name() -> str:
        return PROVIDER_NAME

    @staticmethod
    def get_api_key_name() -> str:
        return API_KEY_NAME

    def _call_sync(self) -> Callable:
        return self._client_sync.messages.create

    def _call_async(self) -> Callable:
        return self._client_async.messages.create

    def _create_client_sync(self, base_url: str, api_key: str):
        return Anthropic(
            base_url=base_url,
            api_key=api_key,
            max_retries=0,
            http_client=Client(
                limits=Limits(
                    max_connections=MAX_CONNECTIONS_PER_PROCESS,
                    max_keepalive_connections=MAX_KEEPALIVE_CONNECTIONS_PER_PROCESS,
                ),
                timeout=Timeout(timeout=HTTP_TIMEOUT, connect=HTTP_CONNECT_TIMEOUT),
            ),
        )

    def _create_client_async(self, base_url: str, api_key: str):
        return AsyncAnthropic(
            base_url=base_url,
            api_key=api_key,
            max_retries=0,
            http_client=AsyncClient(
                limits=Limits(
                    max_connections=MAX_CONNECTIONS_PER_PROCESS,
                    max_keepalive_connections=MAX_KEEPALIVE_CONNECTIONS_PER_PROCESS,
                ),
                timeout=Timeout(timeout=HTTP_TIMEOUT, connect=HTTP_CONNECT_TIMEOUT),
            ),
        )

    def _adjust_temperature(self, temperature: float) -> float:
        return temperature / 2

    def _extract_response(
        self, request: Conversation, response: Message
    ) -> AdapterChatCompletion:
        finish_reason = FINISH_REASON_MAPPING.get(
            AnthropicFinishReason(response.stop_reason), AdapterFinishReason.stop
        )

        choices: list[Choice] = []
        for content in response.content:
            if content.type == "text":
                choices.append(
                    Choice(
                        index=len(choices),
                        finish_reason=finish_reason.value,
                        message=ChatCompletionMessage(
                            role=ConversationRole.assistant.value,
                            content=content.text,
                        ),
                    )
                )
            elif content.type == "tool_use":
                choices.append(
                    Choice(
                        index=len(choices),
                        finish_reason=finish_reason.value,
                        message=ChatCompletionMessage(
                            role=ConversationRole.assistant.value,
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
            # Deprecated
            response=Turn(
                role=ConversationRole.assistant,
                content=choices[0].message.content or "",
            ),
            token_counts=Cost(
                prompt=usage.prompt_tokens,
                completion=usage.completion_tokens,
                request=self.get_model().cost.request,
            ),
        )

    # TODO: add streaming tools support
    def _extract_stream_response(
        self, request, response: RawMessageStreamEvent, state: dict
    ) -> AdapterChatCompletionChunk:
        choice_chunk = ChoiceChunk(
            index=0,
            delta=ChoiceDelta(role=ConversationRole.assistant.value, content=""),
        )

        if isinstance(response, RawMessageStartEvent):
            state["id"] = response.message.id
            state["created"] = int(time.time())
        elif isinstance(response, RawContentBlockDeltaEvent) and isinstance(
            response.delta, TextDelta
        ):
            choice_chunk.delta.content = response.delta.text
        elif isinstance(response, RawMessageDeltaEvent) and response.delta.stop_reason:
            choice_chunk.finish_reason = FINISH_REASON_MAPPING.get(
                AnthropicFinishReason(response.delta.stop_reason),
                AdapterFinishReason.stop,
            ).value

        return AdapterChatCompletionChunk(
            id=state["id"],
            choices=[choice_chunk],
            created=state["created"],
            model=self.get_model().name,
            object="chat.completion.chunk",
        )

    def _get_params(self, llm_input: Conversation, **kwargs) -> Dict[str, Any]:
        params = super()._get_params(llm_input, **kwargs)

        # messages = cast(List[Choice], params["messages"])
        messages = params["messages"]
        system_prompt: Optional[str] = None

        # Extract system prompt if it's the first message
        if len(messages) > 0 and messages[0]["role"] == ConversationRole.system.value:
            system_prompt = messages[0]["content"]
            messages = messages[1:]

        # Remove trailing whitespace from the last assistant message
        if (
            len(messages) > 0
            and messages[-1]["role"] == ConversationRole.assistant.value
        ):
            messages[-1]["content"] = messages[-1]["content"].rstrip()

        # Include base64-encoded images in the request
        for message in messages:
            if (
                isinstance(message["content"], list)
                and message["role"] == ConversationRole.user.value
            ):
                anthropic_content = []

                for content in message["content"]:
                    if content["type"] == "text":
                        anthropic_content.append(content)
                    elif content["type"] == "image_url":
                        anthropic_content.append(
                            process_image_url_anthropic(content["image_url"]["url"])
                        )

                message["content"] = anthropic_content

        # Convert tools to anthropic format
        openai_tools = params.get("tools")
        openai_tools_choice = params.get("tool_choice")

        anthropic_tools: Optional[list[ToolParam]] = None
        anthropic_tool_choice: Optional[ToolChoice] = None

        if openai_tools_choice == "required":
            anthropic_tool_choice = ToolChoiceToolChoiceAny(type="any")
        elif openai_tools_choice == "auto":
            anthropic_tool_choice = ToolChoiceToolChoiceAuto(type="auto")
        elif openai_tools_choice == "none":
            anthropic_tools = None
        elif isinstance(openai_tools_choice, dict):
            anthropic_tool_choice = ToolChoiceToolChoiceTool(
                name=openai_tools_choice["function"]["name"],
                type="tool",
            )

        if openai_tools:
            anthropic_tools = []
            for openai_tool in openai_tools:
                anthropic_tool = ToolParam(
                    name=openai_tool["function"]["name"],
                    description=openai_tool["function"]["description"],
                    input_schema={
                        "type": openai_tool["function"]["parameters"]["type"],
                        "properties": openai_tool["function"]["parameters"][
                            "properties"
                        ],
                        "required": openai_tool["function"]["parameters"]["required"],
                    },
                )

                anthropic_tools.append(anthropic_tool)

        return AnthropicCreate(
            max_tokens=params.get("max_tokens", self.get_model().completion_length),
            messages=messages,
            metadata=params.get("metadata"),
            stop_sequences=params.get("stop_sequences"),
            stream=params.get("stream"),
            system=system_prompt,
            temperature=params.get("temperature"),
            tool_choice=anthropic_tool_choice,
            tools=anthropic_tools,
            top_k=params.get("top_k"),
            top_p=params.get("top_p"),
        ).model_dump()

    def get_base_sdk_url(self) -> str:
        return BASE_URL
