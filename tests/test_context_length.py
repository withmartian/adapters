import pytest

from adapters import AdapterFactory


@pytest.mark.vcr
def test_non_zero_length():
    for model in AdapterFactory.get_supported_models():
        assert model.context_length > 0
