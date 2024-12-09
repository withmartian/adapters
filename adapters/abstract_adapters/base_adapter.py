from abc import ABC, abstractmethod
from typing import Any, Iterable, Literal, Optional, Union, overload

from openai import NOT_GIVEN, NotGiven
from openai.types.chat import ChatCompletionMessageParam

from adapters.types import (
    AdapterChatCompletion,
    AdapterCompletion,
    AdapterStreamAsyncChatCompletion,
    AdapterStreamAsyncCompletion,
    AdapterStreamSyncChatCompletion,
    AdapterStreamSyncCompletion,
    Conversation,
    Model,
    Prompt,
)


class BaseAdapter(ABC):
    def __str__(self) -> str:
        return f"adapter-{self.get_model().get_path()}"

    def set_api_key(self, api_key: str) -> None:
        pass

    @abstractmethod
    def get_model(self) -> Model:
        pass

    @overload
    def execute_sync(
        self,
        llm_input: Iterable[ChatCompletionMessageParam] | Conversation,
        stream: Optional[Literal[False]] | NotGiven = NOT_GIVEN,
        **kwargs: Any,
    ) -> AdapterChatCompletion: ...

    @overload
    def execute_sync(
        self,
        llm_input: Iterable[ChatCompletionMessageParam] | Conversation,
        stream: Literal[True],
        **kwargs: Any,
    ) -> AdapterStreamSyncChatCompletion: ...

    @abstractmethod
    def execute_sync(
        self,
        llm_input: Iterable[ChatCompletionMessageParam] | Conversation,
        stream: Optional[Literal[False]] | Literal[True] | NotGiven = NOT_GIVEN,
        **kwargs: Any,
    ) -> AdapterChatCompletion | AdapterStreamSyncChatCompletion: ...

    @overload
    async def execute_async(
        self,
        llm_input: Iterable[ChatCompletionMessageParam] | Conversation,
        stream: Optional[Literal[False]] | NotGiven = NOT_GIVEN,
        **kwargs: Any,
    ) -> AdapterChatCompletion: ...

    @overload
    async def execute_async(
        self,
        llm_input: Iterable[ChatCompletionMessageParam] | Conversation,
        stream: Literal[True],
        **kwargs: Any,
    ) -> AdapterStreamAsyncChatCompletion: ...

    @abstractmethod
    async def execute_async(
        self,
        llm_input: Iterable[ChatCompletionMessageParam] | Conversation,
        stream: Optional[Literal[False]] | Literal[True] | NotGiven = NOT_GIVEN,
        **kwargs: Any,
    ) -> AdapterChatCompletion | AdapterStreamAsyncChatCompletion: ...

    # Completion
    @overload
    def execute_completion_sync(
        self,
        prompt: Union[str, list[str], Iterable[int], Iterable[Iterable[int]], None],
        stream: Optional[Literal[False]] | NotGiven = NOT_GIVEN,
        **kwargs: Any,
    ) -> AdapterCompletion: ...

    @overload
    def execute_completion_sync(
        self,
        prompt: Union[str, list[str], Iterable[int], Iterable[Iterable[int]], None],
        stream: Literal[True],
        **kwargs: Any,
    ) -> AdapterStreamSyncCompletion: ...

    @abstractmethod
    def execute_completion_sync(
        self,
        prompt: Union[str, list[str], Iterable[int], Iterable[Iterable[int]], None],
        stream: Optional[Literal[False]] | Literal[True] | NotGiven = NOT_GIVEN,
        **kwargs: Any,
    ) -> AdapterCompletion | AdapterStreamSyncCompletion: ...

    @overload
    async def execute_completion_async(
        self,
        prompt: Union[str, list[str], Iterable[int], Iterable[Iterable[int]], None],
        stream: Optional[Literal[False]] | NotGiven = NOT_GIVEN,
        **kwargs: Any,
    ) -> AdapterCompletion: ...

    @overload
    async def execute_completion_async(
        self,
        prompt: Union[str, list[str], Iterable[int], Iterable[Iterable[int]], None],
        stream: Literal[True],
        **kwargs: Any,
    ) -> AdapterStreamAsyncCompletion: ...

    @abstractmethod
    async def execute_completion_async(
        self,
        prompt: Union[str, list[str], Iterable[int], Iterable[Iterable[int]], None],
        stream: Optional[Literal[False]] | Literal[True] | NotGiven = NOT_GIVEN,
        **kwargs: Any,
    ) -> AdapterCompletion | AdapterStreamAsyncCompletion: ...

    # Deprecated
    @staticmethod
    def convert_to_input(llm_input: Conversation | Prompt) -> Conversation:
        if isinstance(llm_input, Conversation):
            return llm_input
        return llm_input.convert_to_conversation()
