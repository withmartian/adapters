from enum import Enum
import time
from typing import Any, Dict

from cohere import AsyncClientV2, ChatResponse, ClientV2
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from adapters.abstract_adapters.sdk_chat_adapter import SDKChatAdapter
from adapters.types import (
    AdapterChatCompletion,
    AdapterFinishReason,
    Conversation,
    ConversationRole,
    Cost,
    Model,
    ModelProperties,
    Provider,
    Turn,
    Vendor,
)


class CohereModel(Model):
    provider_name: str = Provider.cohere.value
    vendor_name: str = Vendor.cohere.value

    properties: ModelProperties = ModelProperties(
        open_source=True, gdpr_compliant=True, is_nsfw=True
    )

    supports_streaming: bool = False
    supports_tools: bool = False

    def _get_api_path(self) -> str:
        return self.name


MODELS: list[Model] = [
    CohereModel(
        name="command-r-plus-08-2024",
        cost=Cost(prompt=2.50e-6, completion=10.00e-6),
        context_length=128000,
    ),
    CohereModel(
        name="command-r-plus-04-2024",
        cost=Cost(prompt=2.50e-6, completion=10.00e-6),
        context_length=128000,
    ),
    CohereModel(
        name="command-r-plus",
        cost=Cost(prompt=2.50e-6, completion=10.00e-6),
        context_length=128000,
    ),
    CohereModel(
        name="command-r-08-2024",
        cost=Cost(prompt=0.15e-6, completion=0.60e-6),
        context_length=128000,
    ),
    CohereModel(
        name="command-r-03-2024",
        cost=Cost(prompt=0.15e-6, completion=0.60e-6),
        context_length=128000,
    ),
    CohereModel(
        name="command-r",
        cost=Cost(prompt=0.15e-6, completion=0.60e-6),
        context_length=128000,
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

    def _create_client_sync(self, base_url: str, api_key: str) -> ClientV2:
        return ClientV2(base_url=base_url, api_key=api_key)  # type: ignore

    def _create_client_async(self, base_url: str, api_key: str) -> AsyncClientV2:
        return AsyncClientV2(base_url=base_url, api_key=api_key)  # type: ignore

    def _get_params(self, llm_input: Conversation, **kwargs: Any) -> dict[str, Any]:
        params = super()._get_params(llm_input, **kwargs)

        messages = []
        for message in params["messages"]:
            new_message = {
                "role": message["role"],
                "content": message["content"],
            }

            if isinstance(new_message["content"], list):
                new_message["content"] = " ".join(
                    content.get("text", "") for content in new_message["content"]
                )

            if new_message["content"] == "":
                new_message["content"] = " "

            messages.append(new_message)

        params["messages"] = messages
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
        else:
            choices.append(
                Choice(
                    index=len(choices),
                    finish_reason=finish_reason.value,
                    message=ChatCompletionMessage(
                        role=ConversationRole.assistant.value,
                        content="",
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
