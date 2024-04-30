import os
from datetime import datetime, timedelta
from typing import Optional

import httpx
from openai import AsyncOpenAI, OpenAI


def log_connection_pool_status(pool):  # pylint: disable=too-many-locals
    # Configuration parameters
    # print(f'pool : {vars(pool)}')
    active_connections = 0
    idle_connections = 0
    connection_status = {}
    available_conn = 0
    expired_conn = 0
    closed_conn = 0
    for conn in pool.connections:
        if not connection_status.get(conn.info(), None):
            connection_status[conn.info()] = 0
        connection_status[conn.info()] += 1
        if conn.is_available():
            available_conn += 1
        if conn.has_expired():
            expired_conn += 1
        if conn.is_closed():
            closed_conn += 1
        if conn.is_idle():
            idle_connections += 1
        else:
            active_connections += 1

    request_conn_idle = 0
    request_conn_active = 0
    request_conn_idle = 0
    request_conn_closed = 0
    request_conn_expired = 0
    request_conn_available = 0
    request_conn_status = {}
    request_without_connection = 0
    for request in pool._requests:
        if request.connection:
            req_conn = request.connection
            if not request_conn_status.get(req_conn.info(), None):
                request_conn_status[req_conn.info()] = 0
            request_conn_status[req_conn.info()] += 1
            if req_conn.is_idle():
                request_conn_idle += 1
            else:
                request_conn_active = +1
            if req_conn.is_available():
                request_conn_available += 1
            if req_conn.is_closed():
                request_conn_closed += 1
            if req_conn.has_expired():
                request_conn_expired += 1
        else:
            request_without_connection += 1

    print(
        "\n\n***************************************************** Pool Info **************************************************************"
    )
    print(
        f"Connections: {len(pool.connections)} (active: {active_connections}, idle: {idle_connections}, available conn : {available_conn}, closed: {closed_conn}, expired: {expired_conn}, connection status: {', '.join([f'`{key}`: {count}' for key, count in connection_status.items()])})"
    )
    print(
        f"Request: , {len(pool._requests)} (without connections: {request_without_connection} ,active: {request_conn_active}, idle: {request_conn_idle}, available conn : {request_conn_available}, closed: {request_conn_closed}, expired: {request_conn_expired}, connection status: {', '.join([f'`{key}`: {count}' for key, count in request_conn_status.items()])})"
    )
    print(
        "***************************************************** END Pool Info **********************************************************\n\n"
    )


LOG_EVERY_N_CLIENT_CREATE = int(os.getenv("LOG_EVERY_N_CLIENT_CREATE", "5"))
LOG_POOL_INFO = os.getenv("LOG_POOL_INFO", "False").lower() in ("true", "1", "t")


class OpenAIClientFactory:
    _http_async_client: httpx.AsyncClient | None = None
    _http_sync_client: httpx.Client | None = None
    _openai_sync_clients: dict[str, OpenAI] = {}
    _openai_async_clients: dict[str, AsyncOpenAI] = {}
    _last_async_client_time: datetime = datetime.now()
    _last_sync_client_time: datetime = datetime.now()
    _instance: Optional["OpenAIClientFactory"] = None
    _count = 0

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls, *args, **kwargs)

        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized"):
            return
        OpenAIClientFactory._http_async_client = httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=int(os.getenv("MAX_CONNECTIONS_PER_PROCESS", "1000")),
                max_keepalive_connections=int(
                    os.getenv("MAX_KEEPALIVE_CONNECTIONS_PER_PROCESS", "100")
                ),
            ),
            timeout=httpx.Timeout(
                timeout=float(os.getenv("HTTPX_TIMEOUT", "600.0")), connect=5.0
            ),
        )

        OpenAIClientFactory._http_sync_client = httpx.Client(
            limits=httpx.Limits(
                max_connections=int(os.getenv("MAX_CONNECTIONS_PER_PROCESS", "1000")),
                max_keepalive_connections=int(
                    os.getenv("MAX_KEEPALIVE_CONNECTIONS_PER_PROCESS", "100")
                ),
            ),
            timeout=httpx.Timeout(
                timeout=float(os.getenv("HTTP_TIMEOUT", "600.0")), connect=5.0
            ),
        )
        self._initialized = True

    @staticmethod
    def get_openai_async_client(api_key: str, base_url: str) -> AsyncOpenAI:
        # Check if 1 hour has passed since the last creation of _http_async_client for the given api_key and base_url
        if (
            datetime.now() - OpenAIClientFactory._last_async_client_time
            > timedelta(
                minutes=float(
                    os.getenv("HTTPX_AND_OPENAI_CLIENT_RESTART_TIME_MINUTES", "30")
                )
            )
            or f"{api_key}-{base_url}" not in OpenAIClientFactory._openai_async_clients
        ):
            OpenAIClientFactory._http_async_client = httpx.AsyncClient(
                limits=httpx.Limits(
                    max_connections=int(
                        os.getenv("MAX_CONNECTIONS_PER_PROCESS", "1000")
                    ),
                    max_keepalive_connections=int(
                        os.getenv("MAX_KEEPALIVE_CONNECTIONS_PER_PROCESS", "100")
                    ),
                ),
                timeout=httpx.Timeout(
                    timeout=float(os.getenv("HTTPX_TIMEOUT", "600.0")), connect=5.0
                ),
            )
            OpenAIClientFactory._last_async_client_time = datetime.now()

            # Create a new AsyncOpenAI client for the given api_key and base_url
            OpenAIClientFactory._openai_async_clients[
                f"{api_key}-{base_url}"
            ] = AsyncOpenAI(
                http_client=OpenAIClientFactory._http_async_client,
                base_url=base_url,
                api_key=api_key,
            )
        if (
            LOG_POOL_INFO
            and OpenAIClientFactory._count % LOG_EVERY_N_CLIENT_CREATE == 0
        ):
            log_connection_pool_status(
                OpenAIClientFactory._openai_async_clients[  # type: ignore
                    f"{api_key}-{base_url}"
                ]._client._transport._pool
            )

        return OpenAIClientFactory._openai_async_clients[f"{api_key}-{base_url}"]

    @staticmethod
    def get_openai_sync_client(api_key: str, base_url: str) -> OpenAI:
        # Check if 1 hour has passed since the last creation of _http_sync_client
        if (
            datetime.now() - OpenAIClientFactory._last_async_client_time
            > timedelta(
                minutes=float(
                    os.getenv("HTTPX_AND_OPENAI_CLIENT_RESTART_TIME_MINUTES", "30")
                )
            )
            or f"{api_key}-{base_url}" not in OpenAIClientFactory._openai_async_clients
        ):
            OpenAIClientFactory._http_sync_client = httpx.Client(
                limits=httpx.Limits(
                    max_connections=int(
                        os.getenv("MAX_CONNECTIONS_PER_PROCESS", "1000")
                    ),
                    max_keepalive_connections=int(
                        os.getenv("MAX_KEEPALIVE_CONNECTIONS_PER_PROCESS", "100")
                    ),
                ),
                timeout=httpx.Timeout(
                    timeout=float(os.getenv("HTTP_TIMEOUT", "600.0")), connect=5.0
                ),
            )
            OpenAIClientFactory._last_sync_client_time = datetime.now()

            OpenAIClientFactory._openai_sync_clients[f"{api_key}-{base_url}"] = OpenAI(
                http_client=OpenAIClientFactory._http_sync_client,
                base_url=base_url,
                api_key=api_key,
            )
        return OpenAIClientFactory._openai_sync_clients[f"{api_key}-{base_url}"]
