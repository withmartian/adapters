from abc import abstractmethod
from typing import Dict, Generic, Optional, Tuple, Union

import adapters.utils.network_utils as network_utils
from adapters.abstract_adapters.api_key_adapter_mixin import ApiKeyAdapterMixin
from adapters.rate_limiter import AbstractRateLimiter
from adapters.types import (
    AdapterResponse,
    AdapterStreamResponse,
    Cost,
    LLMAsyncStreamOutputType,
    LLMInputType,
    LLMOutputType,
    LLMStreamOutputType,
    RequestBody,
    RequestQueryParams,
    ResponseBody,
)
from adapters.utils.general_utils import delete_none_values


class HttpApiAdapterMixin(
    ApiKeyAdapterMixin,
    Generic[LLMInputType, LLMOutputType, LLMStreamOutputType, LLMAsyncStreamOutputType],
):
    """Base class for all Adapters that will send a request over the internet

    Args:
        BaseAdapter (_type_): _description_
    """

    def __init__(self):
        self.method = "POST"
        ApiKeyAdapterMixin.__init__(self)
        super().__init__()

    def set_method(self, method: str) -> None:
        """sets the http method for the adapter

        Args:
            method (str): http method for the adapter
        """
        self.method = method

    def build_request(
        self,
        llm_input: LLMInputType,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> Union[Tuple[RequestBody, None], Tuple[None, RequestQueryParams]]:
        """build request body or query params based on method

        RequestBody: a request body and query params that can be sent to the vendor api
        """
        if self.method == "POST":
            request_body = self._get_request_body(
                llm_input,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            return request_body, None
        if self.method == "GET":
            request_params = self._get_query_params(llm_input, **kwargs)
            return None, request_params

        raise ValueError(f"method {self.method} is not supported")

    async def execute_async(
        self,
        llm_input: LLMInputType,
        **kwargs,
    ) -> LLMOutputType | AdapterStreamResponse[LLMAsyncStreamOutputType]:
        """executes a conversation again the model api asynchronously

        Args:
            messages (Conversation): the list of conversations in json dict format { "role": "user", "content": "hello"}

        Returns:
            ChatAdapterResponse: a ChatAdapterResponse from the model
        """
        if kwargs.get("stream"):
            raise ValueError("streaming is not supported for http api adapters")

        request_body, query_params = self.build_request(
            llm_input,
            **kwargs,
        )

        assert request_body is None or query_params is None

        # OPENAI API doenst like when some fields are present at the same time even if they are null, so we do clean up
        response = await network_utils.async_send_request(
            method=self.method,
            url=self._get_url(),
            data=(
                RequestBody(delete_none_values(dict(request_body)))
                if request_body
                else None
            ),
            headers=self._get_headers(),
            query_params=query_params,
        )

        return self.extract_response(
            request=(
                query_params  # type: ignore[arg-type]
                if request_body is None
                else request_body
            ),
            response=response,
        )

    def execute_sync(
        self,
        llm_input: LLMInputType,
        **kwargs,
    ) -> LLMOutputType | AdapterStreamResponse[LLMStreamOutputType]:
        """same like async but synchronous, sends a conversation to an LLM api and returns the response

        Args:
            messages (Conversation): the list of conversations in json dict format { "role": "user", "content": "hello"}

        Returns:
            ChatAdapterResponse: _description_
        """

        if kwargs.get("stream"):
            raise ValueError("streaming is not supported for http api adapters")

        request_body, query_params = self.build_request(
            llm_input,
            **kwargs,
        )

        assert request_body is None or query_params is None

        # OPENAI API doenst like when some fields are present at the same time even if they are null, so we do clean up
        response = network_utils.send_request(
            url=self._get_url(),
            data=(
                RequestBody(delete_none_values(dict(request_body)))
                if request_body
                else None
            ),
            headers=self._get_headers(),
            query_params=query_params,
            method=self.method,
        )

        return self.extract_response(
            request=(
                query_params  # type: ignore[arg-type]
                if request_body is None
                else request_body
            ),
            response=response,
        )

    @abstractmethod
    def _get_response_body(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        text_response: str,
        token_cost: Cost = Cost(prompt=0, completion=0),
    ) -> AdapterResponse:
        pass

    def get_rate_limiter(self) -> Optional[AbstractRateLimiter]:
        """returns a rate limiter for the adapter

        Returns:
            Optional[AbstractRateLimiter]: rate limiter for the adapter, used in async calls
        """
        return None

    @abstractmethod
    def _get_headers(self) -> Dict[str, str]:
        """returns the headers that are sent on each api call to the vendor

        Returns:
            Dict[str, str]: list of http headers to be send to vendor api
        """

    @abstractmethod
    def _get_url(self) -> str:
        """url of vendor model api

        Returns:
            str: valid vendor url for model
        """

    @abstractmethod
    def _get_request_body(
        self,
        llm_input: LLMInputType,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> RequestBody:
        """converts a input to a request body

        Args:
            input (InputType): input to model (e.g. conversation, prompt, etc.)

        Returns:
            RequestBody: a request body that can be sent to the vendor api
        """

    def _get_query_params(
        self,
        llm_input: LLMInputType,  # pylint: disable=unused-argument
        **kwargs,  # pylint: disable=unused-argument
    ) -> RequestQueryParams:
        """converts a input to a request query params"""
        return RequestQueryParams({})

    @abstractmethod
    def extract_response(
        self,
        request: Union[RequestBody, RequestQueryParams],
        response: ResponseBody,
    ) -> LLMOutputType:
        """extract the response from the vendor api response and sent request body

        Args:
            request_body (RequestBody): http request_body that is sent
            response_body (ResponseBody): http response body that is received from the vendor

        Returns:

            OutputType: output extracted from the vendor (i.e. ChatAdapterResponse, etc.)
        """
