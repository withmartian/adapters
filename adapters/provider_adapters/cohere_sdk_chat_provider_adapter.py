from enum import Enum
import time
from typing import Any, Dict

from cohere import (
    AsyncClientV2,
    ChatContentDeltaEventDelta,
    ChatContentDeltaEventDeltaMessage,
    ChatContentDeltaEventDeltaMessageContent,
    ChatMessageEndEventDelta,
    ChatResponse,
    ClientV2,
    ContentDeltaStreamedChatResponseV2,
    MessageEndStreamedChatResponseV2,
    MessageStartStreamedChatResponseV2,
    StreamedChatResponseV2,
)
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import Choice as ChoiceChunk, ChoiceDelta

from adapters.abstract_adapters.sdk_chat_adapter import SDKChatAdapter
from adapters.types import (
    AdapterChatCompletion,
    AdapterChatCompletionChunk,
    AdapterFinishReason,
    ConversationRole,
    Cost,
    Model,
    Provider,
    Turn,
    Vendor,
)
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)

from openai.types.chat import ChatCompletionMessageParam


class CohereModel(Model):
    provider_name: str = Provider.cohere.value
    vendor_name: str = Vendor.cohere.value

    supports_completion: bool = False
    supports_n: bool = False
    supports_vision: bool = False
    supports_tools_choice: bool = False
    supports_max_completion_tokens: bool = False
    supports_stop: bool = False

    can_empty_content: bool = False
    can_system_only: bool = False
    can_min_p: bool = False
    can_top_p: bool = False
    can_top_k: bool = False

    def _get_api_path(self) -> str:
        return self.name


MODELS: list[Model] = [
    CohereModel(
        name="command-r-plus-04-2024",
        cost=Cost(prompt=3.00e-6, completion=15.00e-6),
        context_length=128000,
        completion_length=4000,
    ),
    CohereModel(
        name="command-r-plus-08-2024",
        cost=Cost(prompt=2.50e-6, completion=10.00e-6),
        context_length=128000,
        completion_length=4000,
    ),
    CohereModel(
        name="command-r-plus",
        cost=Cost(prompt=2.50e-6, completion=10.00e-6),
        context_length=128000,
        completion_length=4000,
    ),
    CohereModel(
        name="command-r-03-2024",
        cost=Cost(prompt=0.50e-6, completion=1.50e-6),
        context_length=128000,
        completion_length=4000,
    ),
    CohereModel(
        name="command-r-08-2024",
        cost=Cost(prompt=0.15e-6, completion=0.60e-6),
        context_length=128000,
        completion_length=4000,
    ),
    CohereModel(
        name="command-r",
        cost=Cost(prompt=0.15e-6, completion=0.60e-6),
        context_length=128000,
        completion_length=4000,
    ),
    CohereModel(
        name="command",
        cost=Cost(prompt=1.00e-6, completion=2.00e-6),
        context_length=4000,
        completion_length=4000,
        supports_json_output=False,
        supports_tools=False,
    ),
    CohereModel(
        name="command-nightly",
        cost=Cost(prompt=1.00e-6, completion=2.00e-6),
        context_length=128000,
        completion_length=128000,
    ),
    CohereModel(
        name="command-light",
        cost=Cost(prompt=0.30e-6, completion=0.60e-6),
        context_length=4000,
        completion_length=4000,
        supports_json_output=False,
        supports_tools=False,
    ),
    CohereModel(
        name="command-light-nightly",
        cost=Cost(prompt=0.30e-6, completion=0.60e-6),
        context_length=4000,
        completion_length=4000,
        supports_json_output=False,
        supports_tools=False,
    ),
    CohereModel(
        name="c4ai-aya-expanse-8b",
        cost=Cost(prompt=0.50e-6, completion=1.50e-6),
        context_length=8000,
        completion_length=4000,
        supports_json_output=False,
        supports_tools=False,
    ),
    CohereModel(
        name="c4ai-aya-expanse-32b",
        cost=Cost(prompt=0.50e-6, completion=1.50e-6),
        context_length=128000,
        completion_length=4000,
        supports_json_output=False,
        supports_tools=False,
    ),
]


class CohereFinishReason(str, Enum):
    complete = "COMPLETE"
    max_tokens = "MAX_TOKENS"
    stop_sequence = "STOP_SEQUENCE"
    tool_call = "TOOL_CALL"
    error = "ERROR"


FINISH_REASON_MAPPING: Dict[CohereFinishReason, AdapterFinishReason] = {
    CohereFinishReason.complete: AdapterFinishReason.stop,
    CohereFinishReason.max_tokens: AdapterFinishReason.length,
    CohereFinishReason.stop_sequence: AdapterFinishReason.stop,
    CohereFinishReason.tool_call: AdapterFinishReason.tool_calls,
}


class CohereSDKChatProviderAdapter(SDKChatAdapter[ClientV2, AsyncClientV2]):
    @staticmethod
    def get_supported_models() -> list[Model]:
        return MODELS

    @staticmethod
    def get_api_key_name() -> str:
        return "COHERE_API_KEY"

    def get_base_sdk_url(self) -> str:
        return "https://api.cohere.com"

    def _sync_client_wrapper(self, **kwargs: Any) -> Any:
        if kwargs.pop("stream", False):
            return self._client_sync.chat_stream(**kwargs)
        return self._client_sync.chat(**kwargs)

    async def _async_client_wrapper(self, **kwargs: Any) -> Any:
        if kwargs.pop("stream", False):
            return self._client_async.chat_stream(**kwargs)
        return await self._client_async.chat(**kwargs)

    def _call_async(self) -> Any:
        return self._async_client_wrapper

    def _call_sync(self) -> Any:
        return self._sync_client_wrapper

    def _call_completion_sync(self) -> Any:
        raise NotImplementedError

    def _call_completion_async(self) -> Any:
        raise NotImplementedError

    def _create_client_sync(self, base_url: str, api_key: str) -> ClientV2:
        return ClientV2(base_url=base_url, api_key=api_key)  # type: ignore

    def _create_client_async(self, base_url: str, api_key: str) -> AsyncClientV2:
        return AsyncClientV2(base_url=base_url, api_key=api_key)  # type: ignore

    def _adjust_temperature(self, temperature: float) -> float:
        return temperature / 2

    def _get_params(
        self, messages: list[ChatCompletionMessageParam], **kwargs: Any
    ) -> dict[str, Any]:
        params = super()._get_params(messages, **kwargs)

        # Cohere does not support extra_body
        params = {**params, **params["extra_body"]}
        del params["extra_body"]

        return params

    def _extract_response(
        self, request: Any, response: ChatResponse
    ) -> AdapterChatCompletion:
        prompt_tokens = int(
            response.usage.billed_units.input_tokens
            if response.usage
            and response.usage.billed_units
            and response.usage.billed_units.input_tokens
            else 0
        )
        completion_tokens = int(
            response.usage.billed_units.output_tokens
            if response.usage
            and response.usage.billed_units
            and response.usage.billed_units.output_tokens
            else 0
        )

        cost = (
            self.get_model().cost.prompt * prompt_tokens
            + self.get_model().cost.completion * completion_tokens
            + self.get_model().cost.request
        )

        finish_reason = FINISH_REASON_MAPPING.get(
            CohereFinishReason(response.finish_reason), AdapterFinishReason.stop
        )

        choices: list[Choice] = []
        if response.message and response.message.content:
            for content in response.message.content:
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
        elif response.message and response.message.tool_calls:
            if response.message.tool_plan:
                choices.append(
                    Choice(
                        index=len(choices),
                        finish_reason=finish_reason.value,
                        message=ChatCompletionMessage(
                            role=ConversationRole.assistant.value,
                            content=response.message.tool_plan,
                        ),
                    )
                )

            for tool_call in response.message.tool_calls:
                if (
                    not tool_call.id
                    or not tool_call.function
                    or not tool_call.function.name
                    or not tool_call.function.arguments
                ):
                    raise ValueError("Unsupported response")

                choices.append(
                    Choice(
                        index=len(choices),
                        finish_reason=finish_reason.value,
                        message=ChatCompletionMessage(
                            role=ConversationRole.assistant.value,
                            tool_calls=[
                                ChatCompletionMessageToolCall(
                                    id=tool_call.id,
                                    type="function",
                                    function=Function(
                                        name=tool_call.function.name,
                                        arguments=tool_call.function.arguments,
                                    ),
                                )
                            ],
                        ),
                    )
                )

        usage = CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
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

    def _extract_stream_response(
        self, request: Any, response: StreamedChatResponseV2, state: dict[str, Any]
    ) -> AdapterChatCompletionChunk:
        choice_chunk = ChoiceChunk(
            index=0,
            delta=ChoiceDelta(role=ConversationRole.assistant.value, content=""),
        )

        # TODO: Add citation support
        if isinstance(response, MessageStartStreamedChatResponseV2):
            state["id"] = response.id
            state["created"] = int(time.time())
        elif (
            isinstance(response, ContentDeltaStreamedChatResponseV2)
            and isinstance(response.delta, ChatContentDeltaEventDelta)
            and isinstance(response.delta.message, ChatContentDeltaEventDeltaMessage)
            and isinstance(
                response.delta.message.content, ChatContentDeltaEventDeltaMessageContent
            )
            and isinstance(response.delta.message.content.text, str)
        ):
            choice_chunk.delta.content = response.delta.message.content.text
        elif isinstance(response, MessageEndStreamedChatResponseV2):
            choice_chunk.finish_reason = AdapterFinishReason.stop.value

            if isinstance(response.delta, ChatMessageEndEventDelta) and isinstance(
                response.delta.finish_reason, ChatMessageEndEventDelta
            ):
                choice_chunk.finish_reason = FINISH_REASON_MAPPING.get(
                    CohereFinishReason(response.delta.finish_reason),
                    AdapterFinishReason.stop,
                ).value
        # else:
        # raise ValueError("Unsupported response")

        return AdapterChatCompletionChunk(
            id=state["id"],
            choices=[choice_chunk],
            created=state["created"],
            model=self.get_model().name,
            object="chat.completion.chunk",
        )

    def _extract_completion_response(
        self,
        request: Any,
        response: Any,
    ) -> Any:
        raise NotImplementedError

    def _extract_completion_stream_response(
        self, request: Any, response: Any, state: dict[str, Any]
    ) -> Any:
        raise NotImplementedError
