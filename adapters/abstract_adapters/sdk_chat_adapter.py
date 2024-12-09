from abc import abstractmethod
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    Generator,
    Generic,
    Iterable,
    Literal,
    Optional,
    TypeVar,
    Union,
    overload,
)

from openai.types.chat import ChatCompletionMessageParam, ChatCompletionUserMessageParam
from openai import NOT_GIVEN, NotGiven

from adapters.abstract_adapters.api_key_adapter_mixin import ApiKeyAdapterMixin
from adapters.abstract_adapters.base_adapter import BaseAdapter
from adapters.abstract_adapters.provider_adapter_mixin import ProviderAdapterMixin
from adapters.client_cache import client_cache
from adapters.constants import OVERRIDE_ALL_BASE_URLS
from adapters.general_utils import (
    EMPTY_CONTENT,
    delete_none_values,
    stream_generator_auto_close,
)
from adapters.types import (
    AdapterChatCompletion,
    AdapterChatCompletionChunk,
    AdapterCompletion,
    AdapterCompletionChunk,
    AdapterException,
    AdapterStreamAsyncChatCompletion,
    AdapterStreamAsyncCompletion,
    AdapterStreamSyncChatCompletion,
    AdapterStreamSyncCompletion,
    ContentType,
    Conversation,
    ConversationRole,
    Model,
    ModelProperties,
)

CLIENT_SYNC = TypeVar("CLIENT_SYNC")
CLIENT_ASYNC = TypeVar("CLIENT_ASYNC")


class SDKChatAdapter(
    BaseAdapter,
    ApiKeyAdapterMixin,
    ProviderAdapterMixin,
    Generic[CLIENT_SYNC, CLIENT_ASYNC],
):
    _client_sync: CLIENT_SYNC
    _client_async: CLIENT_ASYNC

    def __init__(
        self,
    ) -> None:
        super().__init__()
        self._setup_clients(self.get_api_key())

    def _get_or_create_client(
        self, api_key: str, client_type: Literal["sync", "async"]
    ) -> Any:
        base_url = OVERRIDE_ALL_BASE_URLS or self.get_base_sdk_url()
        client = client_cache.get_client(base_url, api_key, client_type)
        if not client:
            client = getattr(self, f"_create_client_{client_type}")(
                api_key=api_key, base_url=base_url
            )
            client_cache.set_client(base_url, api_key, client_type, client)
        return client

    def _setup_clients(self, api_key: str) -> None:
        self._client_sync = self._get_or_create_client(api_key, "sync")
        self._client_async = self._get_or_create_client(api_key, "async")

    @abstractmethod
    def _call_sync(self) -> Callable[..., Any]:
        pass

    @abstractmethod
    def _call_async(self) -> Callable[..., Any]:
        pass

    @abstractmethod
    def _call_completion_sync(self) -> Callable[..., Any]:
        pass

    @abstractmethod
    def _call_completion_async(self) -> Callable[..., Any]:
        pass

    @abstractmethod
    def _create_client_sync(self, base_url: str, api_key: str) -> CLIENT_SYNC:
        pass

    @abstractmethod
    def _create_client_async(self, base_url: str, api_key: str) -> CLIENT_ASYNC:
        pass

    @abstractmethod
    def get_base_sdk_url(self) -> str:
        pass

    @abstractmethod
    def _extract_response(self, request: Any, response: Any) -> AdapterChatCompletion:
        pass

    @abstractmethod
    def _extract_completion_response(
        self, request: Any, response: Any
    ) -> AdapterCompletion:
        pass

    @abstractmethod
    def _extract_stream_response(
        self, request: Any, response: Any, state: dict[str, Any]
    ) -> AdapterChatCompletionChunk:
        pass

    @abstractmethod
    def _extract_completion_stream_response(
        self, request: Any, response: Any, state: dict[str, Any]
    ) -> AdapterCompletionChunk:
        pass

    def _adjust_temperature(self, temperature: float) -> float:
        return temperature

    # TODO: Add tests for exceptions
    def _verify(
        self, messages: list[ChatCompletionMessageParam], **kwargs: Any
    ) -> None:
        if not self.get_model().supports_tools and "tools" in kwargs:
            raise AdapterException(f"Tools is not supported on {self.get_model().name}")

        if (
            not self.get_model().supports_tools_choice
            and kwargs.get("tool_choices") is not None
        ):
            raise AdapterException(
                f"Tool choice is not supported on {self.get_model().name}"
            )

        if (
            kwargs.get("max_tokens")
            and self.get_model().completion_length
            and kwargs.get("max_tokens", 0) > self.get_model().completion_length
        ):
            raise AdapterException(
                f"max_tokens {kwargs.get('max_tokens')} should be less than max completion length {self.get_model().completion_length} for {self.get_model().name}"
            )

        if not self.get_model().supports_streaming and kwargs.get("stream") is True:
            raise AdapterException(
                f"Streaming is not supported on {self.get_model().name}"
            )

        if not self.get_model().supports_functions and "functions" in kwargs:
            raise AdapterException(
                f"Function calling is not supported on {self.get_model().name}"
            )

        if self.get_model().supports_n is False and "n" in kwargs and kwargs["n"] >= 1:
            if kwargs["n"] == 1:
                del kwargs["n"]
            else:
                raise AdapterException(f"n is not supported on {self.get_model().name}")

        if not self.get_model().supports_json_content:
            for message in messages:
                if not isinstance(message["content"], str):
                    raise AdapterException(
                        f"JSON content is not supported on {self.get_model().name}"
                    )

        if not self.get_model().supports_vision:
            for message in messages:
                if not isinstance(message["content"], str) and message["content"]:
                    for content in message["content"]:
                        if (
                            not isinstance(content, str)
                            and content["type"] == ContentType.image_url.value
                        ):
                            raise AdapterException(
                                f"Vision is not supported on {self.get_model().name}"
                            )

        if (
            not self.get_model().supports_json_output
            and "response_format" in kwargs
            and "type" in kwargs["response_format"]
            and kwargs["response_format"]["type"] == "json_object"
        ):
            raise AdapterException(
                f"JSON response format is not supported on {self.get_model().name}"
            )

    # TODO: Refactor this big method
    # TODO: Check if a "system" message is between two "user" messages
    # pylint: disable=too-many-statements
    def _get_params(
        self, messages: list[ChatCompletionMessageParam], **kwargs: Any
    ) -> Dict[str, Any]:
        self._verify(messages, **kwargs)

        # Check if messages are empty
        if len(messages) == 0:
            messages.append(
                ChatCompletionUserMessageParam(
                    role=ConversationRole.user.value, content=EMPTY_CONTENT
                )
            )

        # Convert empty string to EMPTY_CONTENT if not supported
        if not self.get_model().can_empty_content:
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

        # Remove user if not supported
        if not self.get_model().can_user and "user" in kwargs:
            del kwargs["user"]

        # Remove temperature if not supported
        if (
            kwargs.get("temperature") is not None
            and self.get_model().can_temperature is False
        ):
            del kwargs["temperature"]

        # Add empty user message if assistant only
        if (
            not self.get_model().can_assistant_only
            and messages[0]["role"] == ConversationRole.assistant.value
        ):
            messages.append(
                ChatCompletionUserMessageParam(
                    role=ConversationRole.user.value, content=EMPTY_CONTENT
                )
            )

        # Add empty user message if system only
        if (
            not self.get_model().can_system_only
            and messages[0]["role"] == ConversationRole.system.value
        ):
            messages.append(
                ChatCompletionUserMessageParam(
                    role=ConversationRole.user.value, content=EMPTY_CONTENT
                )
            )

        # Change system prompt roles to user if not supported
        if not self.get_model().can_system_multiple:
            for messageId, message in enumerate(messages[1:], start=1):
                if message["role"] == ConversationRole.system.value:
                    messages[messageId] = ChatCompletionUserMessageParam(
                        role=ConversationRole.user.value, content=message["content"]
                    )

        # Change system prompt roles to user
        if not self.get_model().can_system:
            for messageId, message in enumerate(messages):
                if message["role"] == ConversationRole.system.value:
                    messages[messageId] = ChatCompletionUserMessageParam(
                        role=ConversationRole.user.value, content=message["content"]
                    )

        # If the first message is assistant, insert an empty user message
        if (
            self.get_model().can_assistant_first is False
            and messages[0]["role"] == ConversationRole.assistant.value
        ):
            messages.insert(
                0,
                ChatCompletionUserMessageParam(
                    role=ConversationRole.user.value, content=EMPTY_CONTENT
                ),
            )

        # If the last message is assistant, add an empty user message
        if (
            self.get_model().can_assistant_last is False
            and messages[-1]["role"] == ConversationRole.assistant.value
        ):
            messages.append(
                ChatCompletionUserMessageParam(
                    role=ConversationRole.user.value, content=EMPTY_CONTENT
                )
            )

        # If the last message is system, add an empty user message
        if (
            self.get_model().can_system_last is False
            and messages[-1]["role"] == ConversationRole.system.value
        ):
            messages.append(
                ChatCompletionUserMessageParam(
                    role=ConversationRole.user.value, content=EMPTY_CONTENT
                )
            )

        # Join messages from the same role
        if not self.get_model().can_repeating_roles:
            result: list[ChatCompletionMessageParam] = []
            grouped_message: Optional[ChatCompletionMessageParam] = None

            for message in messages:
                if grouped_message and message["role"] == grouped_message["role"]:
                    if isinstance(grouped_message["content"], str) and isinstance(
                        message["content"], str
                    ):
                        grouped_message["content"] = (
                            f"{grouped_message['content']}\n{message['content']}"
                        )
                    elif isinstance(grouped_message["content"], list) and isinstance(
                        message["content"], list
                    ):
                        grouped_message["content"].extend(message["content"])
                    elif isinstance(grouped_message["content"], list) and isinstance(
                        message["content"], str
                    ):
                        grouped_message["content"].append(
                            {"type": ContentType.text.value, "text": message["content"]}
                        )
                    elif isinstance(grouped_message["content"], str) and isinstance(
                        message["content"], list
                    ):
                        grouped_message["content"] = [
                            {
                                "type": ContentType.text.value,
                                "text": grouped_message["content"],
                            }
                        ]  # type: ignore
                        grouped_message["content"].extend(message["content"])  # type: ignore
                else:
                    if grouped_message:
                        result.append(grouped_message)
                    grouped_message = message

            if grouped_message:
                result.append(grouped_message)

            messages = result

        return {
            "messages": messages,
            **(
                {"temperature": self._adjust_temperature(kwargs.get("temperature", 1))}
                if kwargs.get("temperature") is not None
                else {}
            ),
            **kwargs,
        }

        # ====

        return {
            "messages": messages,
            **(
                {"temperature": self._adjust_temperature(kwargs.get("temperature", 1))}
                if kwargs.get("temperature") is not None
                else {}
            ),
            **kwargs,
        }

    def get_model(self) -> Model:
        if self._current_model is None:
            raise ValueError("Model is not set")
        return self._current_model

    def get_model_properteis(self, model_name: str) -> ModelProperties:
        for model in self.get_supported_models():
            if model.name == model_name:
                return model.properties
        raise ValueError(f"Model {model_name} not found")

    def set_api_key(self, api_key: str) -> None:
        super().set_api_key(api_key)
        self._setup_clients(api_key)

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

    async def execute_async(
        self,
        llm_input: Iterable[ChatCompletionMessageParam] | Conversation,
        stream: Optional[Literal[False]] | Literal[True] | NotGiven = NOT_GIVEN,
        **kwargs: Any,
    ) -> AdapterChatCompletion | AdapterStreamAsyncChatCompletion:
        messages = list(
            llm_input.convert_to_openai_format()
            if isinstance(llm_input, Conversation)
            else llm_input
        )

        params = self._get_params(messages, stream=stream, **kwargs)

        response = await self._call_async()(
            model=self.get_model()._get_api_path(),
            **delete_none_values(params),
        )

        if not stream:
            return self._extract_response(request=llm_input, response=response)

        async def stream_response() -> AsyncGenerator[AdapterChatCompletionChunk, None]:
            state: dict[str, Any] = {}
            async with stream_generator_auto_close(response):
                try:
                    async for chunk in response:
                        yield self._extract_stream_response(
                            request=llm_input, response=chunk, state=state
                        )
                except Exception as e:
                    raise AdapterException(f"Error in streaming response: {e}") from e
                finally:
                    if hasattr(response, "close"):
                        await response.close()
                    # elif hasattr(response, "aclose"):  # For Cohere SDK
                    # await response.aclose()

        return AdapterStreamAsyncChatCompletion(response=stream_response())

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

    def execute_sync(
        self,
        llm_input: Iterable[ChatCompletionMessageParam] | Conversation,
        stream: Optional[Literal[False]] | Literal[True] | NotGiven = NOT_GIVEN,
        **kwargs: Any,
    ) -> AdapterChatCompletion | AdapterStreamSyncChatCompletion:
        messages = list(
            llm_input.convert_to_openai_format()
            if isinstance(llm_input, Conversation)
            else llm_input
        )

        params = self._get_params(messages, stream=stream, **kwargs)

        response = self._call_sync()(
            model=self.get_model()._get_api_path(),
            **delete_none_values(params),
        )

        if not stream:
            return self._extract_response(request=llm_input, response=response)

        def stream_response() -> Generator[AdapterChatCompletionChunk, Any, None]:
            state: dict[str, Any] = {}
            try:
                for chunk in response:
                    yield self._extract_stream_response(
                        request=llm_input, response=chunk, state=state
                    )
            except Exception as e:
                raise AdapterException(f"Error in streaming response: {e}") from e
            finally:
                response.close()

        return AdapterStreamSyncChatCompletion(response=stream_response())

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

    def execute_completion_sync(
        self,
        prompt: Union[str, list[str], Iterable[int], Iterable[Iterable[int]], None],
        stream: Optional[Literal[False]] | Literal[True] | NotGiven = NOT_GIVEN,
        **kwargs: Any,
    ) -> AdapterCompletion | AdapterStreamSyncCompletion:
        response = self._call_completion_sync()(
            model=self.get_model()._get_api_path(),
            prompt=prompt,
            **delete_none_values(kwargs),
        )

        if not stream:
            return self._extract_completion_response(request=prompt, response=response)

        def stream_response() -> Generator[AdapterCompletionChunk, Any, None]:
            state: dict[str, Any] = {}
            try:
                for chunk in response:
                    yield self._extract_completion_stream_response(
                        request=prompt, response=chunk, state=state
                    )
            except Exception as e:
                raise AdapterException(f"Error in streaming response: {e}") from e
            finally:
                response.close()

        return AdapterStreamSyncCompletion(response=stream_response())

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

    async def execute_completion_async(
        self,
        prompt: Union[str, list[str], Iterable[int], Iterable[Iterable[int]], None],
        stream: Optional[Literal[False]] | Literal[True] | NotGiven = NOT_GIVEN,
        **kwargs: Any,
    ) -> AdapterCompletion | AdapterStreamAsyncCompletion:
        response = await self._call_completion_async()(
            model=self.get_model()._get_api_path(),
            prompt=prompt,
            **delete_none_values(kwargs),
        )

        if not stream:
            return self._extract_completion_response(request=prompt, response=response)

        async def stream_response() -> AsyncGenerator[AdapterCompletionChunk, None]:
            state: dict[str, Any] = {}
            try:
                async for chunk in response:
                    yield self._extract_completion_stream_response(
                        request=prompt, response=chunk, state=state
                    )
            except Exception as e:
                raise AdapterException(f"Error in streaming response: {e}") from e
            finally:
                response.close()

        return AdapterStreamAsyncCompletion(response=stream_response())
