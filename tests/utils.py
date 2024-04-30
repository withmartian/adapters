import hashlib
import inspect
import json
import os
import re

import requests
import tiktoken_ext.openai_public  # type: ignore

from adapters.types import (
    ContentTurn,
    ContentType,
    Conversation,
    ConversationRole,
    TextContentEntry,
    Turn,
)

ASYNC_LIMITER_LEAK_BUCKET_TIME = 2


TEST_TEMPERATURE = 0.5
TEST_MAX_TOKENS = 200
TEST_TOP_P = 0.5


SIMPLE_CONVERSATION_USER_ONLY = Conversation(
    [Turn(role=ConversationRole.user, content="Hi")]
)

SIMPLE_CONVERSATION_EMPTY_CONTENT = Conversation(
    [
        Turn(role=ConversationRole.user, content=""),
        Turn(role=ConversationRole.assistant, content=" "),
        Turn(role=ConversationRole.user, content="\n"),
    ]
)

SIMPLE_CONVERSATION_TRAILING_WHITESPACE = Conversation(
    [
        Turn(role=ConversationRole.user, content="Hi"),
        Turn(role=ConversationRole.assistant, content="Hi "),
    ]
)

SIMPLE_CONVERSATION_ASSISTANT_SYSTEM = Conversation(
    [
        Turn(role=ConversationRole.user, content="Hi"),
        Turn(role=ConversationRole.assistant, content="Hi"),
        Turn(role=ConversationRole.system, content="Hi"),
        Turn(role=ConversationRole.user, content="Hi"),
    ]
)

SIMPLE_CONVERSATION_ASSISTANT_FIRST = Conversation(
    [
        Turn(role=ConversationRole.assistant, content="Hi"),
        Turn(role=ConversationRole.user, content="Hi"),
    ]
)

SIMPLE_CONVERSATION_MULTIPLE_SYSTEM = Conversation(
    [
        Turn(role=ConversationRole.system, content="Hi"),
        Turn(role=ConversationRole.user, content="Hi"),
        Turn(role=ConversationRole.system, content="Hi"),
    ]
)

SIMPLE_CONVERSATION_REPEATING = Conversation(
    [
        Turn(role=ConversationRole.user, content="Hi"),
        Turn(role=ConversationRole.assistant, content="Hi"),
        Turn(role=ConversationRole.assistant, content="Hi"),
        Turn(role=ConversationRole.user, content="Hi"),
        Turn(role=ConversationRole.user, content="Hi"),
    ]
)

SIMPLE_CONVERSATION_JSON = Conversation(
    [Turn(role=ConversationRole.user, content="Hi, use json")]
)

SIMPLE_CONVERSATION_JSON_CONTENT = Conversation(
    [
        ContentTurn(
            role=ConversationRole.user,
            content=[TextContentEntry(type=ContentType.text, text="Hi")],
        ),
        ContentTurn(
            role=ConversationRole.user,
            content=[TextContentEntry(type=ContentType.text, text="Test")],
        ),
    ]
)


SIMPLE_FUNCTION_CALL_USER_ONLY = Conversation(
    [Turn(role=ConversationRole.user, content="Generate random number")]
)

SIMPLE_CONVERSATION_YOU_RAG_QUESTION = Conversation(
    [Turn(role=ConversationRole.user, content="What is in an egg?")]
)

SIMPLE_CONVERSATION_VISION = Conversation(
    [
        ContentTurn.model_validate(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Hi",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                        },
                    },
                ],
            },
            from_attributes=True,
        )
    ]
)


def setup_tiktoken_cache():
    # print(dir(tiktoken_ext.openai_public))
    # The encoder we want is cl100k_base, we see this as a possible function

    tiktoken_inspect = inspect.getsource(tiktoken_ext.openai_public.cl100k_base)
    # pylint: disable=anomalous-backslash-in-string
    file_location = (
        re.search(
            "(?P<url>https?://[^\s]+)",
            tiktoken_inspect,
        )
        .group("url")
        .replace('"', "")
    )
    # pylint: enable=anomalous-backslash-in-string
    filename = hashlib.sha1(file_location.encode()).hexdigest()
    r = requests.get(file_location, allow_redirects=True)

    open(filename, "wb").write(r.content)
    folder = os.getcwd()
    os.environ["TIKTOKEN_CACHE_DIR"] = folder

    # The URL should be in the 'load_tiktoken_bpe function call'


def get_choices_from_vcr(vcr):
    last_response = vcr.responses[len(vcr.responses) - 1]
    choices = json.loads(
        last_response["content"]
        if "content" in last_response
        else last_response["body"]["string"]
    )["choices"]
    return choices
