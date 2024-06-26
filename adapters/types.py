from enum import Enum
from typing import Any, Dict, Generic, List, Literal, Optional, TypeVar, Union

from openai.types.chat import ChatCompletionMessageToolCall
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import FunctionCall
from pydantic import BaseModel


class ResponseBody(Dict[str, Any]):
    pass


class RequestBody(Dict[str, Any]):
    pass


class RequestQueryParams(Dict[str, str]):
    pass


LLMInputType = TypeVar("LLMInputType")
LLMOutputType = TypeVar("LLMOutputType")
LLMStreamOutputType = TypeVar("LLMStreamOutputType")
LLMAsyncStreamOutputType = TypeVar("LLMAsyncStreamOutputType")

LLMSyncClientType = TypeVar("LLMSyncClientType")
LLMAsyncClientType = TypeVar("LLMAsyncClientType")

AdapterStreamResponseType = TypeVar("AdapterStreamResponseType")

ResponseSDKType = TypeVar("ResponseSDKType")


class ConversationRole(str, Enum):
    user = "user"
    assistant = "assistant"
    system = "system"
    function = "function"
    tool = "tool"


class FinishReason(str, Enum):
    stop = "stop"
    length = "length"
    tool_calls = "tool_calls"
    content_filter = "content_filter"
    function_call = "function_call"
    stop_sequence = "stop_sequence"


class Turn(BaseModel, use_enum_values=True):
    role: ConversationRole
    content: str


class FunctionOutputTurn(BaseModel, use_enum_values=True):
    role: Literal[ConversationRole.function] = ConversationRole.function
    content: Optional[str] = None
    name: str


class ToolOutputTurn(BaseModel, use_enum_values=True):
    role: Literal[ConversationRole.tool] = ConversationRole.tool
    content: Optional[str] = None
    tool_call_id: str


class FunctionCallTurn(BaseModel, use_enum_values=True):
    role: Literal[ConversationRole.assistant] = ConversationRole.assistant
    function_call: FunctionCall
    content: None = None


class ToolsCallTurn(BaseModel, use_enum_values=True):
    role: Literal[ConversationRole.assistant] = ConversationRole.assistant
    tool_calls: list[ChatCompletionMessageToolCall]
    content: None = None


class ImageDetailsType(str, Enum):
    high = "high"
    low = "low"
    auto = "auto"


class ContentType(str, Enum):
    text = "text"
    image_url = "image_url"


class VisionImageDetails(BaseModel, use_enum_values=True):
    url: str
    details: Optional[ImageDetailsType] = ImageDetailsType.auto


class TextContentEntry(BaseModel, use_enum_values=True):
    type: Literal[ContentType.text] = ContentType.text
    text: str


class ImageContentEntry(BaseModel, use_enum_values=True):
    type: Literal[ContentType.image_url] = ContentType.image_url
    image_url: VisionImageDetails


class ContentTurn(BaseModel, use_enum_values=True, validate_assignment=True):
    role: Literal[ConversationRole.user] = ConversationRole.user
    content: List[Union[ImageContentEntry, TextContentEntry]]


class Cost(BaseModel):
    prompt: float
    completion: float
    request: float = 0.0


# Add test cases for all of them
class Model(BaseModel):
    _test_async: bool = True

    name: str
    vendor_name: str
    provider_name: str
    cost: Cost
    context_length: int
    supports_repeating_roles: bool = True
    supports_streaming: bool = False
    supports_vision: bool = False
    supports_functions: bool = False
    supports_tools: bool = False
    supports_n: bool = False
    supports_system: bool = True
    supports_multiple_system: bool = True
    supports_empty_content: bool = True
    supports_json_output: bool = False
    supports_json_content: bool = False
    supports_last_assistant: bool = True
    supports_first_assistant: bool = True
    completion_length: Optional[int] = None

    def get_path(self) -> str:
        return f"{self.provider_name}/{self.vendor_name}/{self.name}"

    def _get_api_path(self) -> str:
        return self.name


class Conversation(BaseModel):
    turns: List[
        Union[Turn, FunctionOutputTurn, ToolOutputTurn, ToolsCallTurn, ContentTurn]
    ]

    def __init__(
        self,
        turns: Union[
            List[
                Union[
                    Turn,
                    FunctionOutputTurn,
                    ToolOutputTurn,
                    ToolsCallTurn,
                    ContentTurn,
                ]
            ],
            "Conversation",
            Dict[
                str,
                List[
                    Union[
                        Turn,
                        FunctionOutputTurn,
                        ToolOutputTurn,
                        ToolsCallTurn,
                        ContentTurn,
                    ]
                ],
            ],
        ],
    ):
        if isinstance(turns, Conversation):
            turns = turns.turns
        elif isinstance(turns, dict) and "turns" in turns:
            turns = turns["turns"]
        super().__init__(turns=turns)

    def __getitem__(self, index):
        return self.turns[index]

    def __setitem__(self, index, value):
        if not isinstance(value, Turn):
            raise ValueError("Value must be an instance of Turn")
        self.turns[index] = value

    def __len__(self):
        return len(self.turns)

    def __iter__(self):
        return iter(self.turns)

    def is_last_turn_vision_query(self) -> bool:
        if len(self.turns):
            contentTurn = self.turns[len(self.turns) - 1]
        else:
            return False

        if isinstance(contentTurn, ContentTurn):
            return any(
                content.type == ContentType.image_url for content in contentTurn.content
            )

        return False

    def convert_to_prompt(self) -> "Prompt":
        return Prompt("".join([f"{turn.role}: {turn.content}" for turn in self.turns]))

    def convert_to_anthropic_prompt(self) -> "Prompt":
        role_map = {
            "user": "Human",
            "assistant": "Assistant",
            "system": "Human: This is your system prompt. This prompt defines your behavior and personality: \n",
        }
        return Prompt(
            "\n\n".join(
                [""]
                + [
                    f"{role_map.get(turn.role, turn.role)}: {turn.content}"
                    for turn in self.turns
                ]
            )
            + "\n\nAssistant:"
        )


class Prompt(str):
    def convert_to_conversation(self) -> Conversation:
        return Conversation(turns=[Turn(role=ConversationRole.user, content=self)])


class AdapterResponse(BaseModel, Generic[LLMOutputType]):
    response: LLMOutputType
    token_counts: Optional[Cost] = None
    cost: float


class AdapterStreamResponse(BaseModel, Generic[AdapterStreamResponseType]):
    response: AdapterStreamResponseType


class ChatAdapterResponse(AdapterResponse[Turn]):
    pass


class OpenAIChatAdapterResponse(
    AdapterResponse[Union[Turn, FunctionCallTurn, ToolsCallTurn]]
):
    finish_reason: Optional[str] = None
    choices: Optional[List[Dict[str, Any]]] | Optional[List[Choice]] = None


class AdapterException(Exception):
    pass


class AdapterRateLimitException(AdapterException):
    pass


class YouComRagChatAdapterHitEntry(BaseModel):
    ai_snippets: str
    description: str
    snippet: str
    title: str
    url: str


class YouComRagChatAdapterResponse(ChatAdapterResponse):
    hits: List[YouComRagChatAdapterHitEntry]
    latency: float
