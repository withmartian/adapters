import pytest
from pydantic import BaseModel

from adapters.types import (
    ChatAdapterResponse,
    ContentTurn,
    Conversation,
    ConversationRole,
    Prompt,
    Turn,
)

USER_CONTENT = "Hello World, what do you say?"
ASSISTANT_CONTENT = "I say hello world to you too"
SYSTEM_CONTENT = "You are a friendly person that likes to make conversation"


VISION_CALL_TURN_DICT = {
    "role": "user",
    "content": [
        {
            "type": "text",
            "text": "What are in these images? Is there any difference between them?",
        },
        {
            "type": "image_url",
            "image_url": {
                "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
            },
        },
        {
            "type": "image_url",
            "image_url": {
                "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
            },
        },
    ],
}


def test_chat_adapter_response_create_ok():
    content = "Hello, world!"
    turn = Turn(content=content, role=ConversationRole.assistant)
    response = ChatAdapterResponse(
        response=turn,
        token_counts={"prompt": 1.0, "completion": 1.0},
        cost=1.0,
    )
    assert response.response == turn


def test_turn_with_invalid_role_fails():
    content = "Hello, world!"
    with pytest.raises(Exception):
        Turn(content=content, role="balbla")


def test_prompt_to_conversation_ok():
    content = "Hello, world!"
    prompt = Prompt(content)
    conversation = prompt.convert_to_conversation()
    assert len(conversation) == 1
    assert conversation[0].content == content
    assert conversation[0].role == ConversationRole.user


def test_conversation_to_prompt_ok():
    content = "Hello, world!"
    conversation = Conversation([Turn(content=content, role=ConversationRole.user)])
    prompt = conversation.convert_to_prompt()
    assert prompt == f"{ConversationRole.user.value}: {content}"


def test_conversation_to_anthropic_prompt_ok():
    content = "Hello, world!"
    conversation = Conversation([Turn(content=content, role=ConversationRole.user)])
    prompt = conversation.convert_to_anthropic_prompt()
    assert prompt == "\n\nHuman: Hello, world!\n\nAssistant:"


def test_conversation_is_pydantic_model():
    assert issubclass(Conversation, BaseModel)


def test_conversation_can_take_turns_or_list_as_constructor():
    # Test the old input format
    old_input = [Turn(role="user", content="How many toes does a dog have?")]
    conversation_old = Conversation(old_input)

    # Test the new input format
    new_input = {
        "turns": [{"role": "user", "content": "How many toes does a dog have?"}]
    }
    conversation_new = Conversation(**new_input)
    conversation_new2 = Conversation(new_input)

    assert conversation_new == conversation_old  # Should be converted to the old format
    assert conversation_new == conversation_new2


def remove_detail_keys(d):
    if isinstance(d, dict):
        return {k: remove_detail_keys(v) for k, v in d.items() if k != "details"}
    elif isinstance(d, list):
        return [remove_detail_keys(item) for item in d]
    else:
        return d


def test_conversation_image_type_ok():
    vision_call_turn = ContentTurn.model_validate(
        VISION_CALL_TURN_DICT, from_attributes=True
    )
    vision_call_turn_dict_without_detail = remove_detail_keys(
        vision_call_turn.model_dump()
    )
    assert vision_call_turn_dict_without_detail == VISION_CALL_TURN_DICT
