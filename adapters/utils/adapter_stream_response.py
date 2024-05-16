from typing import Any


class stream_generator_auto_close:
    _agen: Any

    def __init__(self, agen):
        self._agen = agen

    async def __aenter__(self):
        return self._agen

    async def __aexit__(self, *args):
        if getattr(self._agen, "close", False):
            await self._agen.close()
