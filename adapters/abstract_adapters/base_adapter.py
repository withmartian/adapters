from abc import ABC, abstractmethod
from typing import Generic

from adapters.types import (
    AdapterStreamResponse,
    Conversation,
    LLMAsyncStreamOutputType,
    LLMInputType,
    LLMOutputType,
    LLMStreamOutputType,
    Model,
    Prompt,
)


class BaseAdapter(
    Generic[LLMInputType, LLMOutputType, LLMStreamOutputType, LLMAsyncStreamOutputType],
    ABC,
):
    """
    Abstract base class for all LLM adapters.
    It defines the interface for creating subclasses and concrete adapters.
    Provides the structure to create a nesting of different adapters and LLM access
    """

    @abstractmethod
    def get_model(self) -> Model:
        """Returns the martian model object, this is used to identify the model which is used by the adapter

        Returns:
            # Model: model object
        """

    @abstractmethod
    async def execute_async(
        self, llm_input: LLMInputType, **kwargs
    ) -> LLMOutputType | AdapterStreamResponse[LLMAsyncStreamOutputType]:
        """Run an LLM request against the model asynchronously over http

        Args:
            llm_input (LLMInputType): input to the LLM model (e.g. conversation, prompt, etc.)

        Returns:
            LLMOutputType: output from the LLM (i.e. ChatAdapterResponse, etc.)
        """

    @abstractmethod
    def execute_sync(
        self, llm_input: LLMInputType, **kwargs
    ) -> LLMOutputType | AdapterStreamResponse[LLMStreamOutputType]:
        """Runs an LLM request against the model synchronously over https

        Args:
            llm_input (LLMInputType): input to the LLM model api (e.g. conversation, prompt, etc.)

        Returns:
            LLMOutputType: output from the LLM api (i.e. ChatAdapterResponse, etc.)
        """

    @staticmethod
    @abstractmethod
    def convert_to_input(llm_input: Conversation | Prompt) -> LLMInputType:
        """converts any input to a valid LLM input

        Args:
            Any: any input

        Returns:
            LLMInputType: valid LLM input
        """

    def adjust_temperature(self, temperature: float) -> float:
        return temperature
