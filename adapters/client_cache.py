from typing import Any

# key: base_url-api_key
# value: function that calls client
_client_cache: dict[str, Any] = {}
