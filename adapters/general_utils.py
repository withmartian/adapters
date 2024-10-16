import base64
import os
from typing import Any

import httpx

from adapters.types import Cost

EMPTY_CONTENT = '""'


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


def process_image_url_anthropic(image_url: str):
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


class stream_generator_auto_close:
    _agen: Any

    def __init__(self, agen):
        self._agen = agen

    async def __aenter__(self):
        return self._agen

    async def __aexit__(self, *args):
        if getattr(self._agen, "close", False):
            await self._agen.close()
