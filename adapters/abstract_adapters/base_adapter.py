from abc import ABC, abstractmethod
from typing import Generic, Optional

from adapters.types import (
    AdapterStreamResponse,
    Conversation,
    Cost,
    LLMAsyncStreamOutputType,
    LLMInputType,
    LLMOutputType,
    LLMStreamOutputType,
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
    def get_name(self) -> str:
        """returns the martain adapter name, this is used to identify the adapter in the martain API

        Returns:
            # str: vendor name / (optional type of adapter) / model name (e.g. openai/gpt-4)
        """

    @abstractmethod
    def get_model_name(self) -> str:
        """returns the vendor model name

        Returns:
            str: provider_id + model_id (e.g. openai/gpt-4)
        """

    @abstractmethod
    def get_context_length(self) -> int:
        """returns the context length of the model

        Returns:
            int: context length
        """

    def get_completion_length(self) -> Optional[int]:
        """returns the completion length of the model

        Returns:
            int: completion length
        """

        return None

    @abstractmethod
    def get_token_cost(self) -> Cost:
        """returns the token cost

        Returns:
            TokenCost: token cost structure_
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

    def supports_tools(self) -> bool:
        return False

    def supports_functions(self) -> bool:
        return False

    def supports_vision(self) -> bool:
        return False

    def supports_streaming(self) -> bool:
        return False

    def supports_n(self) -> bool:
        return False

    def supports_json_output(self) -> bool:
        return False

    def supports_json_content(self) -> bool:
        return False

    def adjust_temperature(self, temperature: float) -> float:
        return temperature
