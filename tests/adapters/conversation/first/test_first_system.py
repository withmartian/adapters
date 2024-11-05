import pytest

from adapters.abstract_adapters.base_adapter import BaseAdapter
from adapters.types import Conversation, ConversationRole, Turn
from tests.utils import (
    TEST_ADAPTERS,
    get_response_content_from_vcr,
)
from vcr import VCR


@pytest.mark.parametrize("adapter", TEST_ADAPTERS)
@pytest.mark.vcr
async def test_async(vcr: VCR, adapter: BaseAdapter) -> None:
    conversation = Conversation(
        [
            Turn(role=ConversationRole.system, content="Hi"),
            Turn(role=ConversationRole.user, content="Hi"),
        ]
    )

    adapter_response = await adapter.execute_async(conversation)

    cassette_response = get_response_content_from_vcr(vcr, adapter)

    assert adapter_response.choices[0].message.content == cassette_response
