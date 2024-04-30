from typing import Tuple

import pytest

from adapters.types import Conversation, ConversationRole, RequestBody, Turn
from adapters.utils.general_utils import load_request_body_with_additional_params
from tests.utils import TEST_MAX_TOKENS, TEST_TEMPERATURE, TEST_TOP_P


def get_new_request_body() -> Tuple[RequestBody, RequestBody]:
    ret_val = RequestBody(
        {
            "input": "My little Song",
            "conversation_input": Conversation(
                turns=[Turn(content="My little Song", role=ConversationRole.user)]
            ),
        }
    )
    ret_val_copy = ret_val.copy()
    return ret_val, RequestBody(ret_val_copy)


TEST_KWARGS_1 = {"anotherone": "anotherone"}
TEST_KWARGS_2 = {
    "anotherone": "anotherone",
    "temperature": 0.8,
    "max_tokens": 100,
    "top_p": 0.5,
}


def test_load_request_body_with_additional_params_no_params_ok():
    request_body, request_body_loaded = get_new_request_body()
    load_request_body_with_additional_params(request_body_loaded)
    assert len(request_body.keys()) == len(request_body_loaded.keys())


def test_load_request_body_with_additional_params_temperature_ok():
    request_body, request_body_loaded = get_new_request_body()
    load_request_body_with_additional_params(
        request_body_loaded, temperature=TEST_TEMPERATURE
    )
    assert len(request_body.keys()) + 1 == len(request_body_loaded.keys())
    assert request_body_loaded["temperature"] == TEST_TEMPERATURE


def test_load_request_body_with_additional_params_temperature_and_max_tokens_ok():
    request_body, request_body_loaded = get_new_request_body()
    load_request_body_with_additional_params(
        request_body_loaded, temperature=TEST_TEMPERATURE, max_tokens=TEST_MAX_TOKENS
    )
    assert len(request_body.keys()) + 2 == len(request_body_loaded.keys())
    assert request_body_loaded["temperature"] == TEST_TEMPERATURE
    assert request_body_loaded["max_tokens"] == TEST_MAX_TOKENS


def test_load_request_body_with_additional_params_temperature_and_max_tokens_and_top_p_ok():
    request_body, request_body_loaded = get_new_request_body()
    load_request_body_with_additional_params(
        request_body_loaded,
        temperature=TEST_TEMPERATURE,
        max_tokens=TEST_MAX_TOKENS,
        top_p=TEST_TOP_P,
    )
    assert len(request_body.keys()) + 3 == len(request_body_loaded.keys())
    assert request_body_loaded["temperature"] == TEST_TEMPERATURE
    assert request_body_loaded["max_tokens"] == TEST_MAX_TOKENS
    assert request_body_loaded["top_p"] == TEST_TOP_P


def test_load_request_body_with_additional_params_all_params_ok():
    request_body, request_body_loaded = get_new_request_body()

    load_request_body_with_additional_params(
        request_body_loaded,
        temperature=TEST_TEMPERATURE,
        max_tokens=TEST_MAX_TOKENS,
        top_p=TEST_TOP_P,
        **TEST_KWARGS_1
    )
    assert len(request_body.keys()) + 3 + len(TEST_KWARGS_1.keys()) == len(
        request_body_loaded.keys()
    )
    assert request_body_loaded["temperature"] == TEST_TEMPERATURE
    assert request_body_loaded["max_tokens"] == TEST_MAX_TOKENS
    assert request_body_loaded["top_p"] == TEST_TOP_P
    assert request_body_loaded["anotherone"] == TEST_KWARGS_1["anotherone"]


def test_load_request_body_with_overriding_params_in_kwargs_fails():
    _, request_body_loaded = get_new_request_body()
    with pytest.raises(TypeError):
        load_request_body_with_additional_params(  # pylint: disable=repeated-keyword
            request_body_loaded,
            temperature=TEST_TEMPERATURE,
            max_tokens=TEST_MAX_TOKENS,
            top_p=TEST_TOP_P,
            **TEST_KWARGS_2
        )
