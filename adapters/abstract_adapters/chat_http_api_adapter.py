from adapters.abstract_adapters.base_adapter import BaseAdapter
from adapters.types import (
    ChatAdapterResponse,
    ConversationRole,
    Cost,
    LLMInputType,
    Turn,
)

from .http_api_adapter_mixin import HttpApiAdapterMixin


class ChatHttpApiAdapter(
    HttpApiAdapterMixin[LLMInputType, ChatAdapterResponse, None, None],
    BaseAdapter[LLMInputType, ChatAdapterResponse, None, None],
):
    """Base class for all ChatAdapters,

    Args:
        BaseAdapter (_type_): _description_
    """

    def __init__(self):
        super().__init__()
        BaseAdapter.__init__(self)

    def _get_response_body(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        text_response: str,
        token_cost: Cost = Cost(prompt=0, completion=0),
    ) -> ChatAdapterResponse:
        return ChatAdapterResponse(
            response=Turn(
                role=ConversationRole.assistant,
                content=text_response,
            ),
            token_counts=Cost(
                prompt=prompt_tokens,
                completion=completion_tokens,
            ),
            cost=token_cost.prompt * prompt_tokens
            + token_cost.completion * completion_tokens,
        )
