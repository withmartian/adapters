import base64
import os
from typing import Optional

import httpx

from adapters.types import Cost, RequestBody

EMPTY_CONTENT = '""'


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


def process_image_url(image_url: str):
    if image_url.startswith("data:"):
        # Base64 data is passed as a URL
        media_type, _, base64_data = image_url.partition(";base64,")
        media_type = media_type.split(":")[1]
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": base64_data,
            },
        }
    else:
        # URL points to an image file
        image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8")
        _, extension = os.path.splitext(image_url)
        extension = extension.lstrip(".").lower()
        if extension == "jpg":
            extension = "jpeg"
        media_type = f"image/{extension}"
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": image_data,
            },
        }


def get_dynamic_cost(model_name: str, token_count: int) -> Cost:
    if model_name == "gemini-1.0-pro":
        return Cost(prompt=0.5e-6, completion=1.5e-6)
    elif model_name == "gemini-1.5-pro":
        if token_count <= 128000:
            return Cost(prompt=3.5e-6, completion=10.5e-6)
        else:
            return Cost(prompt=7.0e-6, completion=21.0e-6)
    elif model_name == "gemini-1.5-flash":
        if token_count <= 128000:
            return Cost(prompt=0.075e-6, completion=0.30e-6)
        else:
            return Cost(prompt=0.15e-6, completion=0.60e-6)
    else:
        raise ValueError(f"Unknown model: {model_name}")
