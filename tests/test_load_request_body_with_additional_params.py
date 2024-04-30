from adapters.types import RequestBody
from adapters.utils.general_utils import load_request_body_with_additional_params

SIMPLE_REQUEST_BODY = RequestBody({"model": "test-model", "messages": "hi"})
SAMPLE_TEMPERATURE = 0.7
SAMPLE_MAX_TOKENS = 10
SAMPLE_TOP_P = 1
SAMPLE_TOP_K = 1


def test_load_request_body_with_zero_values_ok():
    request_body = SIMPLE_REQUEST_BODY.copy()
    load_request_body_with_additional_params(
        request_body, temperature=0, max_tokens=0, top_p=0, top_k=0
    )

    assert request_body["temperature"] == 0
    assert request_body["max_tokens"] == 0
    assert request_body["top_p"] == 0
    assert request_body["top_k"] == 0


def test_load_request_body_with_non_zero_values_ok():
    request_body = SIMPLE_REQUEST_BODY.copy()
    load_request_body_with_additional_params(
        request_body,
        temperature=SAMPLE_TEMPERATURE,
        max_tokens=SAMPLE_MAX_TOKENS,
        top_p=SAMPLE_TOP_P,
        top_k=SAMPLE_TOP_K,
    )

    assert request_body["temperature"] == SAMPLE_TEMPERATURE
    assert request_body["max_tokens"] == SAMPLE_MAX_TOKENS
    assert request_body["top_p"] == SAMPLE_TOP_P
    assert request_body["top_k"] == SAMPLE_TOP_K


def test_load_request_body_with_no_params_ok():
    request_body = SIMPLE_REQUEST_BODY.copy()
    load_request_body_with_additional_params(request_body)

    assert request_body == SIMPLE_REQUEST_BODY
