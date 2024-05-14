"""
You.COM RAG Chat Adapter, is slightly different to most LLM Models we use.
There is not an exact token usages returned it has a fixed cost of 0.0049
There are not temperature and other LLM like settings, as what is does it uses LLM to sum a list of sources.
It also returns not just generated text but also a list of sources that it used to generate the text.
Those are included in the adapter response that is derived from ChatAdapterResponse if that is ever needed.
"""

import re
from typing import Dict, Pattern, Union

from adapters.abstract_adapters.chat_http_api_adapter import ChatHttpApiAdapter
from adapters.types import (
    Conversation,
    ConversationRole,
    Cost,
    Model,
    Prompt,
    RequestBody,
    RequestQueryParams,
    ResponseBody,
    Turn,
    YouComRagChatAdapterHitEntry,
    YouComRagChatAdapterResponse,
)

YOU_COM_RAG_API_KEY_NAME = "YOU_COM_RAG_API_KEY"
YOU_COM_RAG_API_KEY_HEADER_NAME = "X-API-Key"
MODEL_NAME = "rag"
CONTEXT_LENGTH = 4096
CALL_FIXED_COST = 0.0049

REG_API_URL = "https://api.ydc-index.io/rag"

API_KEY_PATTERN = re.compile(
    r"^[a-zA-Z0-9]{8}-[a-zA-Z0-9]{4}-[a-zA-Z0-9]{4}-[a-zA-Z0-9]{4}-[a-zA-Z0-9]{12}<__>[a-zA-Z0-9]+$"
)


YOU_COM_MODEL = Model(
    vendor_name="you",
    provider_name="you",
    name=MODEL_NAME,
    cost=Cost(prompt=1.95e-6, completion=2.6e-6, request=CALL_FIXED_COST),
    context_length=CONTEXT_LENGTH,
)


class YouComRagChatAdapter(ChatHttpApiAdapter[Conversation]):
    def __init__(self):
        super().__init__()
        self.method = "GET"

    def get_context_length(self) -> int:
        return CONTEXT_LENGTH

    @staticmethod
    def get_api_key_name() -> str:
        return YOU_COM_RAG_API_KEY_NAME

    @staticmethod
    def get_api_key_pattern() -> Pattern:
        return API_KEY_PATTERN

    def get_model(self) -> Model:
        return YOU_COM_MODEL

    def _get_headers(self) -> Dict[str, str]:
        """returns the headers that are sent on each api call to the vendor

        Returns:
            Dict[str, str]: list of http headers to be send to vendor api
        """
        return {
            YOU_COM_RAG_API_KEY_HEADER_NAME: self.get_api_key(),
            "Content-Type": "application/json",
        }

    def _get_url(self) -> str:
        return REG_API_URL

    def extract_response(
        self, request: Union[RequestBody, RequestQueryParams], response: ResponseBody
    ) -> YouComRagChatAdapterResponse:
        return YouComRagChatAdapterResponse(
            response=Turn(role=ConversationRole.assistant, content=response["answer"]),
            cost=CALL_FIXED_COST,
            hits=list(
                YouComRagChatAdapterHitEntry(**entry) for entry in response["hits"]
            ),
            latency=response["latency"],
        )

    @staticmethod
    def convert_to_input(llm_input: Conversation | Prompt) -> Conversation:
        # if llm_input is Conversation:
        if isinstance(llm_input, Conversation):
            return llm_input

        if isinstance(llm_input, Prompt):
            return llm_input.convert_to_conversation()

        raise ValueError(f"llm_input {llm_input} is not a valid input")

    def _get_query_params(
        self,
        llm_input: Conversation,
        **kwargs,
    ) -> RequestQueryParams:
        return RequestQueryParams({"query": llm_input[0].content})
