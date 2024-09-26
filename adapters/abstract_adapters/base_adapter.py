from abc import ABC, abstractmethod
from typing import Literal, Optional, overload

from openai import NOT_GIVEN, NotGiven

from adapters.types import (
    AdapterChatCompletion,
    AdapterStreamChatCompletion,
    Conversation,
    Model,
)


class BaseAdapter(ABC):
    @abstractmethod
    def get_model(self) -> Model:
        pass

    @overload
    @abstractmethod
    async def execute_async(
        self,
        llm_input: Conversation,
        stream: Optional[Literal[False]] | NotGiven = NOT_GIVEN,
        **kwargs,
    ) -> AdapterChatCompletion:
        pass

    @overload
    @abstractmethod
    async def execute_async(
        self,
        llm_input: Conversation,
        stream: Optional[Literal[True]] | NotGiven = NOT_GIVEN,
        **kwargs,
    ) -> AdapterStreamChatCompletion:
        pass

    @abstractmethod
    @overload
    def execute_sync(
        self,
        llm_input: Conversation,
        stream: Optional[Literal[False]] | NotGiven = NOT_GIVEN,
        **kwargs,
    ) -> AdapterChatCompletion:
        pass

    @abstractmethod
    @overload
    def execute_sync(
        self,
        llm_input: Conversation,
        stream: Optional[Literal[True]] | NotGiven = NOT_GIVEN,
        **kwargs,
    ) -> AdapterStreamChatCompletion:
        pass
