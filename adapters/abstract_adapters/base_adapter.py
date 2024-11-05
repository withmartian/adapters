from abc import ABC, abstractmethod
from typing import Any, Literal, Optional, overload

from openai import NOT_GIVEN, NotGiven

from adapters.types import (
    AdapterChatCompletion,
    AdapterStreamAsyncChatCompletion,
    AdapterStreamSyncChatCompletion,
    Conversation,
    Model,
    Prompt,
)


class BaseAdapter(ABC):
    def set_api_key(self, api_key: str) -> None:
        pass

    @abstractmethod
    def get_model(self) -> Model:
        pass

    @overload
    def execute_sync(
        self,
        llm_input: Conversation,
        stream: Optional[Literal[False]] | NotGiven = NOT_GIVEN,
        **kwargs: Any,
    ) -> AdapterChatCompletion: ...

    @overload
    def execute_sync(
        self,
        llm_input: Conversation,
        stream: Literal[True],
        **kwargs: Any,
    ) -> AdapterStreamSyncChatCompletion: ...

    @abstractmethod
    def execute_sync(
        self,
        llm_input: Conversation,
        stream: Optional[Literal[False]] | Literal[True] | NotGiven = NOT_GIVEN,
        **kwargs: Any,
    ) -> AdapterChatCompletion | AdapterStreamSyncChatCompletion:
        pass

    @overload
    async def execute_async(
        self,
        llm_input: Conversation,
        stream: Optional[Literal[False]] | NotGiven = NOT_GIVEN,
        **kwargs: Any,
    ) -> AdapterChatCompletion: ...

    @overload
    async def execute_async(
        self,
        llm_input: Conversation,
        stream: Literal[True],
        **kwargs: Any,
    ) -> AdapterStreamAsyncChatCompletion: ...

    @abstractmethod
    async def execute_async(
        self,
        llm_input: Conversation,
        stream: Optional[Literal[False]] | Literal[True] | NotGiven = NOT_GIVEN,
        **kwargs: Any,
    ) -> AdapterChatCompletion | AdapterStreamAsyncChatCompletion:
        pass

    # Deprecated
    @staticmethod
    def convert_to_input(llm_input: Conversation | Prompt) -> Conversation:
        if isinstance(llm_input, Conversation):
            return llm_input
        return llm_input.convert_to_conversation()
