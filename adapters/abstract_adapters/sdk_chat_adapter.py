from abc import abstractmethod
from typing import Any, Dict, Optional

from openai import AsyncStream, Stream
from openai.types.chat import ChatCompletionChunk

from adapters.abstract_adapters.base_adapter import BaseAdapter
from adapters.types import (
    AdapterException,
    AdapterStreamResponse,
    ChatAdapterResponse,
    ContentTurn,
    ContentType,
    Conversation,
    Prompt,
)
from adapters.utils.adapter_stream_response import stream_generator_auto_close
from adapters.utils.general_utils import delete_none_values


class SDKChatAdapter(
    BaseAdapter[
        Conversation,
        ChatAdapterResponse,
        Stream[ChatCompletionChunk],
        AsyncStream[ChatCompletionChunk],
    ],
):
    @staticmethod
    @abstractmethod
    def get_base_sdk_url() -> str:
        pass

    def get_custom_sdk_url(self) -> Optional[str]:
        return None

    @abstractmethod
    def get_async_client(self):
        pass

    @abstractmethod
    def get_sync_client(self):
        pass

    @abstractmethod
    def extract_response(self, request, response):
        pass

    @abstractmethod
    def extract_stream_response(self, request, response):
        pass

    @staticmethod
    def convert_to_input(llm_input: Conversation | Prompt) -> Conversation:
        if isinstance(llm_input, Conversation):
            return llm_input

        if isinstance(llm_input, Prompt):
            return llm_input.convert_to_conversation()

        raise ValueError(f"Llm_input {llm_input} is not a valid input")

    def get_params(
        self,
        llm_input: Conversation,
        **kwargs,  # TODO: type kwargs
    ) -> Dict[str, Any]:
        delete_none_values(kwargs)

        completion_length = self.get_model().completion_length
        if (
            kwargs.get("max_tokens")
            and completion_length
            and kwargs.get("max_tokens", 0) > completion_length
        ):
            raise AdapterException(
                f"max_tokens {kwargs.get('max_tokens')} should be less than max completion length {completion_length} for {self.get_model().name}"
            )

        if (
            self.get_model().supports_streaming is False
            and kwargs.get("stream") is True
        ):
            raise AdapterException(
                f"Streaming is not supported on {self.get_model().name}"
            )

        if self.get_model().supports_functions is False and "functions" in kwargs:
            raise AdapterException(
                f"Function calling is not supported on {self.get_model().name}"
            )

        if self.get_model().supports_tools is False and "tools" in kwargs:
            raise AdapterException(f"Tools is not supported on {self.get_model().name}")

        if self.get_model().supports_n is False and "n" in kwargs and kwargs["n"] >= 1:
            if kwargs["n"] == 1:
                del kwargs["n"]
            else:
                raise AdapterException(f"n is not supported on {self.get_model().name}")

        if self.get_model().supports_vision is False:
            for turn in llm_input.turns:
                if isinstance(turn, ContentTurn):
                    for content in turn.content:
                        if content.type == ContentType.image_url:
                            raise AdapterException(
                                f"Image input is not supported on {self.get_model().name}"
                            )

        if (
            self.get_model().supports_json_output is False
            and "response_format" in kwargs
            and kwargs["response_format"]["type"] == "json_object"
        ):
            raise AdapterException(
                f"JSON response format is not supported on {self.get_model().name}"
            )

        return {
            "messages": [turn.model_dump() for turn in llm_input.turns],
            **(
                {"temperature": self.adjust_temperature(kwargs.get("temperature", 1))}
                if kwargs.get("temperature") is not None
                else {}
            ),
            **kwargs,
        }

    async def execute_async(
        self,
        llm_input: Conversation,
        **kwargs,
    ):
        params = self.get_params(llm_input, **kwargs)

        response = await self.get_async_client()(
            model=self.get_model()._get_api_path(),
            **params,
        )

        if params.get("stream", False):

            async def stream_response():
                async with stream_generator_auto_close(response):
                    try:
                        async for chunk in response:
                            yield self.extract_stream_response(
                                request=llm_input, response=chunk
                            )
                    except Exception as e:
                        raise AdapterException(
                            f"Error in streaming response: {e}"
                        ) from e

            return AdapterStreamResponse(response=stream_response())

        return self.extract_response(request=llm_input, response=response)

    def execute_sync(
        self,
        llm_input: Conversation,
        **kwargs,
    ):
        params = self.get_params(llm_input, **kwargs)

        response = self.get_sync_client()(
            model=self.get_model()._get_api_path(),
            **params,
        )

        if params.get("stream", False):

            def stream_response():
                try:
                    for chunk in response:
                        yield self.extract_stream_response(
                            request=llm_input, response=chunk
                        )
                except Exception as e:
                    raise AdapterException(f"Error in streaming response: {e}") from e
                finally:
                    response.close()

            return AdapterStreamResponse(response=stream_response())

        return self.extract_response(request=llm_input, response=response)
