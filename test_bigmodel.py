from adapters import AdapterFactory
from adapters.types import Conversation, ConversationRole, Turn
from adapters.provider_adapters.bigmodel_provider_adapter import (
    BigModelSDKChatProviderAdapter,
)
import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# Synchronous test
def test_sync() -> None:
    # Debug: Print API key (be careful with this in production!)
    api_key = os.getenv("BIGMODEL_API_KEY")
    if not api_key:
        raise ValueError("BIGMODEL_API_KEY not found in environment variables")
    print("API Key found:", bool(api_key))

    adapter = AdapterFactory.get_adapter_by_path("bigmodel/bigmodel/glm-4-plus")
    if not isinstance(adapter, BigModelSDKChatProviderAdapter):
        raise ValueError("Adapter not found or wrong type")

    conversation = Conversation(
        [
            Turn(
                role=ConversationRole.system, content="You are a helpful AI assistant."
            ),
            Turn(role=ConversationRole.user, content="What is Python?"),
        ]
    )
    response = adapter.execute_sync(conversation)
    print("Sync Response:", response)


# Async test
async def test_async() -> None:
    adapter = AdapterFactory.get_adapter_by_path("bigmodel/bigmodel/glm-4-plus")
    if not isinstance(adapter, BigModelSDKChatProviderAdapter):
        raise ValueError("Adapter not found or wrong type")

    conversation = Conversation(
        [
            Turn(
                role=ConversationRole.system, content="You are a helpful AI assistant."
            ),
            Turn(role=ConversationRole.user, content="What is Python?"),
        ]
    )
    response = await adapter.execute_async(conversation)
    print("Async Response:", response)


if __name__ == "__main__":
    # Run sync test
    test_sync()

    # Run async test
    asyncio.run(test_async())
