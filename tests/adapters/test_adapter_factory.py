from adapters import AdapterFactory


def test_supported_models_length_nonzero() -> None:
    models = AdapterFactory.get_supported_models()
    assert len(models)


def test_supported_models_work() -> None:
    for model in AdapterFactory.get_supported_models():
        assert AdapterFactory.get_adapter_by_path(model.get_path())


def test_non_zero_length() -> None:
    for model in AdapterFactory.get_supported_models():
        assert model.context_length > 0
