import pytest

from adapters import AdapterFactory


@pytest.mark.vcr
def test_non_zero_length() -> None:
    for model in AdapterFactory.get_supported_models():
        assert model.context_length > 0
