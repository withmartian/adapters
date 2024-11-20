from adapters import AdapterFactory
from adapters.types import Conversation, ConversationRole, Turn
from adapters.provider_adapters.bigmodel_provider_adapter import (
    BigModelSDKChatProviderAdapter,
)
import asyncio


# Synchronous test
def test_sync() -> None:
    adapter = AdapterFactory.get_adapter_by_path("bigmodel/bigmodel/glm-4")
    if not isinstance(adapter, BigModelSDKChatProviderAdapter):
        raise ValueError("Adapter not found or wrong type")

    conversation = Conversation(
        [Turn(role=ConversationRole.user, content="What is Python?")]
    )
    response = adapter.execute_sync(conversation)
    print("Sync Response:", response)


# Async test
async def test_async() -> None:
    adapter = AdapterFactory.get_adapter_by_path("bigmodel/bigmodel/glm-4")
    if not isinstance(adapter, BigModelSDKChatProviderAdapter):
        raise ValueError("Adapter not found or wrong type")

    conversation = Conversation(
        [Turn(role=ConversationRole.user, content="What is Python?")]
    )
    response = await adapter.execute_async(conversation)  # Changed to execute_async
    print("Async Response:", response)


if __name__ == "__main__":
    # Run sync test
    test_sync()

    # Run async test
    asyncio.run(test_async())
