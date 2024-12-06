from typing import Any, Callable

from httpx import AsyncClient, Client, Limits, Timeout
from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.completion import Completion

from adapters.abstract_adapters.sdk_chat_adapter import SDKChatAdapter
from adapters.constants import (
    HTTP_CONNECT_TIMEOUT,
    HTTP_TIMEOUT,
    MAX_CONNECTIONS_PER_PROCESS,
    MAX_KEEPALIVE_CONNECTIONS_PER_PROCESS,
)
from adapters.types import (
    AdapterChatCompletion,
    AdapterChatCompletionChunk,
    AdapterCompletion,
    AdapterCompletionChunk,
    ConversationRole,
    Cost,
    Turn,
)


class OpenAISDKChatAdapter(SDKChatAdapter[OpenAI, AsyncOpenAI]):
    def _call_completion_sync(self) -> Callable[..., Any]:
        return self._client_sync.completions.create

    def _call_completion_async(self) -> Callable[..., Any]:
        return self._client_async.completions.create

    def _call_sync(self) -> Callable[..., Any]:
        return self._client_sync.chat.completions.create

    def _call_async(self) -> Callable[..., Any]:
        return self._client_async.chat.completions.create

    def _create_client_sync(self, base_url: str, api_key: str) -> OpenAI:
        return OpenAI(
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

    def _create_client_async(self, base_url: str, api_key: str) -> AsyncOpenAI:
        return AsyncOpenAI(
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

    def _extract_response(
        self,
        request: dict[str, Any],
        response: ChatCompletion,
    ) -> AdapterChatCompletion:
        prompt_tokens = response.usage.prompt_tokens if response.usage else 0
        completion_tokens = response.usage.completion_tokens if response.usage else 0
        reasoning_tokens = (
            response.usage.completion_tokens_details.reasoning_tokens
            if response.usage
            and response.usage.completion_tokens_details
            and response.usage.completion_tokens_details.reasoning_tokens
            else 0
        )

        cost = (
            self.get_model().cost.prompt * prompt_tokens
            + self.get_model().cost.completion * completion_tokens
            + reasoning_tokens * completion_tokens
            + self.get_model().cost.request
        )

        return AdapterChatCompletion.model_construct(
            **response.model_dump(),
            cost=cost,
            # Deprecated
            response=Turn(
                role=ConversationRole.assistant,
                content=response.choices[0].message.content or "",
            ),
            token_counts=Cost(
                prompt=prompt_tokens,
                completion=completion_tokens,
                request=self.get_model().cost.request,
            ),
        )

    def _extract_stream_response(
        self, request: Any, response: ChatCompletionChunk, state: dict[str, Any]
    ) -> AdapterChatCompletionChunk:
        return AdapterChatCompletionChunk.model_construct(
            **response.model_dump(),
        )

    def _extract_completion_response(
        self,
        request: Any,
        response: Completion,
    ) -> AdapterCompletion:
        prompt_tokens = response.usage.prompt_tokens if response.usage else 0
        completion_tokens = response.usage.completion_tokens if response.usage else 0
        reasoning_tokens = (
            response.usage.completion_tokens_details.reasoning_tokens
            if response.usage
            and response.usage.completion_tokens_details
            and response.usage.completion_tokens_details.reasoning_tokens
            else 0
        )

        cost = (
            self.get_model().cost.prompt * prompt_tokens
            + self.get_model().cost.completion * completion_tokens
            + reasoning_tokens * completion_tokens
            + self.get_model().cost.request
        )

        return AdapterCompletion.model_construct(
            **response.model_dump(),
            cost=cost,
        )

    def _extract_completion_stream_response(
        self, request: Any, response: Completion, state: dict[str, Any]
    ) -> AdapterCompletionChunk:
        return AdapterCompletionChunk.model_construct(
            **response.model_dump(),
        )
