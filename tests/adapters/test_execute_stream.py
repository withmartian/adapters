import pytest

from adapters.adapter_factory import AdapterFactory
from tests.adapters.utils.constants import MODEL_PATHS
from tests.utils import (
    SIMPLE_CONVERSATION_USER_ONLY,
)


@pytest.mark.parametrize("model_name", MODEL_PATHS)
@pytest.mark.vcr
def test_sync_execute_streaming(model_name):
    adapter = AdapterFactory.get_adapter_by_path(model_name)

    assert adapter is not None

    if adapter.get_model().supports_streaming is False:
        return

    adapter_response = adapter.execute_sync(
        SIMPLE_CONVERSATION_USER_ONLY,
        stream=True,
    )

    chunks = list(adapter_response.response)

    response = "".join(
        [
            chunk.choices[0].delta.content
            for chunk in chunks
            if chunk.choices[0].delta.content
        ]
    )
    assert len(response) > 0


@pytest.mark.parametrize("model_name", MODEL_PATHS)
@pytest.mark.vcr
async def test_async_execute_streaming(model_name):
    adapter = AdapterFactory.get_adapter_by_path(model_name)

    assert adapter is not None

    if adapter.get_model().supports_streaming is False:
        return

    adapter_response = await adapter.execute_async(
        SIMPLE_CONVERSATION_USER_ONLY,
        stream=True,
    )

    chunks = [data_chunk async for data_chunk in adapter_response.response]

    response = "".join(
        [
            chunk.choices[0].delta.content
            for chunk in chunks
            if chunk.choices[0].delta.content
        ]
    )
    assert len(response) > 0
