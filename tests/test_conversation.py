from adapters.types import Conversation, ConversationRole, Turn


def test_conversation_creation_with_array() -> None:
    conversation = Conversation(
        [Turn(role=ConversationRole.user, content="How many toes does a dog have?")]
    )
    assert conversation.turns[0].role == ConversationRole.user
    assert conversation.turns[0].content == "How many toes does a dog have?"


def test_conversation_creation_with_dict() -> None:
    conversation = Conversation(
        [{"role": "user", "content": "How many toes does a dog have?"}]  # type: ignore
    )
    assert conversation.turns[0].role == ConversationRole.user
    assert conversation.turns[0].content == "How many toes does a dog have?"
