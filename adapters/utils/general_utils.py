import json
from typing import Optional

from adapters.types import RequestBody


def is_json_serializable(obj):
    try:
        json.dumps(obj)
        return True
    except TypeError:
        return False


def load_request_body_with_additional_params(
    request_body: RequestBody,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    **kwargs,
) -> RequestBody:
    for key, value in kwargs.items():
        request_body.setdefault(key, value)
    if temperature is not None:
        request_body.setdefault("temperature", temperature)
    if max_tokens is not None:
        request_body.setdefault("max_tokens", max_tokens)
    return request_body


def delete_none_values(dictionary: dict):
    if isinstance(dictionary, list):
        return [delete_none_values(e) for e in dictionary]

    if isinstance(dictionary, dict):
        for k, v in list(dictionary.items()):
            if v is None:
                del dictionary[k]
            else:
                dictionary[k] = delete_none_values(v)
    return dictionary
