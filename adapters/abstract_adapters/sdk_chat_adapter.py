from abc import abstractmethod
from typing import Any, Dict, List, Optional

from openai import NOT_GIVEN, NotGiven

from adapters.abstract_adapters.base_adapter import BaseAdapter
from adapters.types import (
    AdapterChatCompletion,
    AdapterChatCompletionChunk,
    AdapterException,
    ContentTurn,
    ContentType,
    Conversation,
    ConversationRole,
    Model,
    ModelProperties,
)
from adapters.utils.adapter_stream_response import stream_generator_auto_close
from adapters.utils.general_utils import EMPTY_CONTENT, delete_none_values


class SDKChatAdapter(BaseAdapter):
    @abstractmethod
    def get_supported_models(self) -> List[Model]:
        pass

    @abstractmethod
    def get_base_sdk_url(self) -> str:
        pass

    @abstractmethod
    def get_async_client(self):
        pass

    @abstractmethod
    def get_sync_client(self):
        pass

    @abstractmethod
    def extract_response(self, request, response) -> AdapterChatCompletion:
        pass

    @abstractmethod
    def extract_stream_response(
        self, request, response, state
    ) -> AdapterChatCompletionChunk:
        pass

    def get_model_properteis(self, model_name: str) -> ModelProperties:
        for model in self.get_supported_models():
            if model.name == model_name:
                return model.properties
        raise ValueError(f"Model {model_name} not found")

    def adjust_temperature(self, temperature: float) -> float:
        return temperature

    # pylint: disable=too-many-statements
    def get_params(
        self,
        llm_input: Conversation,
        **kwargs,  # TODO: type kwargs
    ) -> Dict[str, Any]:
        if kwargs.get("stream") == NOT_GIVEN:
            kwargs["stream"] = False

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

        if self.get_model().supports_user is False and "user" in kwargs:
            del kwargs["user"]

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

        messages = [turn.model_dump() for turn in llm_input.turns]

        # ====

        # Convert json content to string if not supported
        if not self.get_model().supports_json_content:
            for message in messages:
                if "content" in message and isinstance(message["content"], list):
                    message["content"] = "\n".join(
                        [
                            content["text"]
                            for content in message["content"]
                            if "text" in content
                        ]
                    )

        # Convert empty string to "." if not supported
        if not self.get_model().supports_empty_content:
            for message in messages:
                if (
                    isinstance(message["content"], str)
                    and message["content"].strip() == ""
                ):
                    message["content"] = EMPTY_CONTENT
                elif isinstance(message["content"], list):
                    for content in message["content"]:
                        if (
                            isinstance(content, dict)
                            and "text" in content
                            and content["text"].strip() == ""
                        ):
                            content["text"] = EMPTY_CONTENT

        # Change system prompt roles to assistant
        if not self.get_model().supports_multiple_system:
            for message in messages[1:]:
                if message["role"] == ConversationRole.system.value:
                    message["role"] = ConversationRole.assistant.value

        # Change system prompt roles to assistant
        if not self.get_model().supports_system:
            for message in messages:
                if message["role"] == ConversationRole.system.value:
                    message["role"] = ConversationRole.assistant.value

        # Join messages from the same role
        processed_messages = []
        current_role = messages[0]["role"]
        current_content = messages[0]["content"]

        for message in messages[1:]:
            if message["role"] == current_role:
                if isinstance(current_content, list) and isinstance(
                    message["content"], list
                ):
                    current_content.extend(message["content"])
                elif isinstance(current_content, list) and isinstance(
                    message["content"], str
                ):
                    current_content.append({"type": "text", "text": message["content"]})
                elif isinstance(current_content, str) and isinstance(
                    message["content"], list
                ):
                    current_content = [
                        {"type": "text", "text": current_content},
                        *message["content"],
                    ]
            else:
                # Otherwise, add the collected messages and reset for the next role
                processed_messages.append(
                    {"role": current_role, "content": current_content}
                )
                current_role = message["role"]
                current_content = message["content"]

        processed_messages.append({"role": current_role, "content": current_content})
        messages = processed_messages

        # If the last message is assistant, add an empty user message
        if (
            self.get_model().supports_first_assistant is False
            and messages[0]["role"] == ConversationRole.assistant.value
        ):
            messages.insert(
                0, {"role": ConversationRole.user.value, "content": EMPTY_CONTENT}
            )

            # If the first message is assistant, add an empty user message
        if (
            self.get_model().supports_last_assistant is False
            and messages[-1]["role"] == ConversationRole.assistant.value
        ):
            messages.append(
                {"role": ConversationRole.user.value, "content": EMPTY_CONTENT}
            )

        # ====

        return {
            "messages": messages,
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
        stream: Optional[bool] | NotGiven = NOT_GIVEN,
        **kwargs,
    ):
        params = self.get_params(llm_input, stream=stream, **kwargs)

        response = await self.get_async_client()(
            model=self.get_model()._get_api_path(),
            **delete_none_values(params),
        )

        if not stream:
            return self.extract_response(request=llm_input, response=response)

        async def stream_response():
            state = {}
            async with stream_generator_auto_close(response):
                try:
                    async for chunk in response:
                        yield self.extract_stream_response(
                            request=llm_input, response=chunk, state=state
                        )
                except Exception as e:
                    raise AdapterException(f"Error in streaming response: {e}") from e
                finally:
                    await response.close()

        return stream_response()

    def execute_sync(
        self,
        llm_input: Conversation,
        stream: Optional[bool] | NotGiven = NOT_GIVEN,
        **kwargs,
    ):
        params = self.get_params(llm_input, stream=stream, **kwargs)

        response = self.get_sync_client()(
            model=self.get_model()._get_api_path(),
            **delete_none_values(params),
        )

        if not stream:
            return self.extract_response(request=llm_input, response=response)

        def stream_response():
            state = {}
            try:
                for chunk in response:
                    yield self.extract_stream_response(
                        request=llm_input, response=chunk, state=state
                    )
            except Exception as e:
                raise AdapterException(f"Error in streaming response: {e}") from e
            finally:
                response.close()

        return stream_response()
