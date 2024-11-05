import pytest

from adapters.abstract_adapters.base_adapter import BaseAdapter
from adapters.types import Conversation, ConversationRole, Turn
from tests.utils import (
    TEST_ADAPTERS,
    get_response_content_from_vcr,
)
from vcr import VCR


conversation = Conversation(
    [
        Turn(role=ConversationRole.system, content=""),
        Turn(role=ConversationRole.user, content=""),
        Turn(role=ConversationRole.assistant, content=" "),
        Turn(role=ConversationRole.user, content="\n"),
    ]
)


@pytest.mark.vcr
@pytest.mark.parametrize("adapter", TEST_ADAPTERS, ids=lambda adapter: str(adapter))
async def test_async(vcr: VCR, adapter: BaseAdapter) -> None:
    adapter_response = await adapter.execute_async(conversation)

    cassette_response = get_response_content_from_vcr(vcr, adapter)

    assert adapter_response.choices[0].message.content == cassette_response
