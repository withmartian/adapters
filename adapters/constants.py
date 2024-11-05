import os

OVERRIDE_ALL_BASE_URLS = os.getenv("_ADAPTERS_OVERRIDE_ALL_BASE_URLS_")

# Has fallback env for backwards compatibility
MAX_CONNECTIONS_PER_PROCESS = int(
    os.getenv(
        "ADAPTERS_MAX_CONNECTIONS_PER_PROCESS",
        os.getenv("MAX_CONNECTIONS_PER_PROCESS", "1000"),
    )
)
MAX_KEEPALIVE_CONNECTIONS_PER_PROCESS = int(
    os.getenv(
        "ADAPTERS_MAX_KEEPALIVE_CONNECTIONS_PER_PROCESS",
        os.getenv("MAX_KEEPALIVE_CONNECTIONS_PER_PROCESS", "100"),
    )
)
HTTP_TIMEOUT = float(
    os.getenv("ADAPTERS_HTTP_TIMEOUT", os.getenv("HTTP_TIMEOUT", "600.0"))
)
HTTP_CONNECT_TIMEOUT = float(
    os.getenv("ADAPTERS_HTTP_CONNECT_TIMEOUT", os.getenv("HTTP_CONNECT_TIMEOUT", "5.0"))
)
