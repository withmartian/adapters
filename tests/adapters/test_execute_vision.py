import pytest

from adapters.adapter_factory import AdapterFactory
from tests.adapters.utils.constants import MODEL_PATHS, MODEL_PATHS_ASYNC
from tests.utils import SIMPLE_CONVERSATION_VISION, get_response_content_from_vcr


@pytest.mark.parametrize("model_path", MODEL_PATHS)
@pytest.mark.vcr
def test_sync(vcr, model_path: str):
    adapter = AdapterFactory.get_adapter_by_path(model_path)

    assert adapter is not None

    if adapter.get_model().supports_vision is False:
        return

    adapter_response = adapter.execute_sync(SIMPLE_CONVERSATION_VISION)

    cassette_response = get_response_content_from_vcr(vcr, adapter)

    assert adapter_response.choices[0].message.content == cassette_response


@pytest.mark.parametrize("model_path", MODEL_PATHS_ASYNC)
@pytest.mark.vcr
async def test_async(vcr, model_path: str):
    adapter = AdapterFactory.get_adapter_by_path(model_path)

    assert adapter is not None

    if adapter.get_model().supports_vision is False:
        return

    adapter_response = await adapter.execute_async(SIMPLE_CONVERSATION_VISION)

    cassette_response = get_response_content_from_vcr(vcr, adapter)

    assert adapter_response.choices[0].message.content == cassette_response
