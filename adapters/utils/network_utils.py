import asyncio
import atexit
import logging
import os
from typing import Dict, Optional

import aiohttp
import requests

from adapters.types import (
    AdapterRateLimitException,
    RequestBody,
    RequestQueryParams,
    ResponseBody,
)

logger = logging.getLogger("adapters.networking")


def raise_exceptions(
    status_code: int,
    method: str,
    url: str,
    data: Optional[RequestBody],
    query_params: Optional[RequestQueryParams],
) -> None:
    # todo: add more exceptions based on the openai and anthropic error codes
    if status_code == 429:
        raise AdapterRateLimitException(
            f"Rate limit exceeded for {url} method {method} with data {data} and query string {query_params}"
        )


_connector = None
_session = None


async def async_send_request(
    url: str,
    method: str,
    headers: Dict,
    data: Optional[RequestBody] = None,
    query_params: Optional[RequestQueryParams] = None,
) -> ResponseBody:
    global _session, _connector  # pylint: disable=global-statement
    if not _connector:
        connections_per_host: int = int(os.getenv("AIOHTTP_LIMIT_PER_HOST", "50"))
        _connector = aiohttp.TCPConnector(limit_per_host=connections_per_host)

    if not _session or _session.closed:
        _session = aiohttp.ClientSession(connector=_connector)

    async with _session.request(
        method=method, url=url, headers=headers, json=data, params=query_params
    ) as response:
        raise_exceptions(
            method=method,
            url=url,
            status_code=response.status,
            data=data,
            query_params=query_params,
        )
        return await response.json()


def send_request(
    url: str,
    method: str,
    headers: Dict,
    data: Optional[RequestBody] = None,
    query_params: Optional[RequestQueryParams] = None,
) -> ResponseBody:
    response = requests.request(
        method=method, url=url, headers=headers, json=data, params=query_params
    )
    raise_exceptions(
        method=method,
        url=url,
        status_code=response.status_code,
        data=data,
        query_params=query_params,
    )
    return response.json()


def close_session_on_exit():
    try:
        loop = asyncio.get_event_loop()
        if _session:
            loop.run_until_complete(_session.close())
    except RuntimeError as e:
        if not str(e).startswith("There is no current event loop in thread"):
            raise e


atexit.register(close_session_on_exit)
