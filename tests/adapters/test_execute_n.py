import pytest

from adapters.adapter_factory import AdapterFactory
from tests.adapters.utils.constants import MODEL_PATHS, N_PARAM
from tests.utils import SIMPLE_CONVERSATION_USER_ONLY, get_response_content_from_vcr


@pytest.mark.parametrize("model_path", MODEL_PATHS)
@pytest.mark.vcr
def test_sync(vcr, model_path: str):
    adapter = AdapterFactory.get_adapter_by_path(model_path)

    assert adapter is not None

    if adapter.get_model().supports_n is False:
        return

    adapter_response = adapter.execute_sync(SIMPLE_CONVERSATION_USER_ONLY, n=N_PARAM)

    cassette_response = get_response_content_from_vcr(vcr, adapter)

    assert adapter_response.choices[0].message.content == cassette_response
    assert len(adapter_response.choices) == N_PARAM


@pytest.mark.parametrize("model_path", MODEL_PATHS)
@pytest.mark.vcr
async def test_async(vcr, model_path: str):
    adapter = AdapterFactory.get_adapter_by_path(model_path)

    assert adapter is not None

    if adapter.get_model().supports_n is False:
        return

    adapter_response = await adapter.execute_async(
        SIMPLE_CONVERSATION_USER_ONLY, n=N_PARAM
    )

    cassette_response = get_response_content_from_vcr(vcr, adapter)

    assert adapter_response.choices[0].message.content == cassette_response
    assert len(adapter_response.choices) == N_PARAM
