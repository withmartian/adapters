import importlib
import os

from adapters import AdapterFactory
from adapters.abstract_adapters.provider_adapter_mixin import ProviderAdapterMixin
from tests.utils import *


# TODO: Add concrete adapter support
def test_get_supported_models_ok():
    models = AdapterFactory.get_supported_models()
    found_models = []

    providers_file_path = os.path.join(
        os.path.dirname(__file__), "../adapters/provider_adapters"
    )
    provider_files = [
        f[:-3]
        for f in os.listdir(providers_file_path)
        if f.endswith(".py") and not f.startswith("__")
    ]

    for f in os.listdir(providers_file_path):
        for module_name in provider_files:
            module = importlib.import_module(
                f"adapters.provider_adapters.{module_name}"
            )
            for _, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and issubclass(obj, ProviderAdapterMixin):
                    if obj.get_supported_models() is None:
                        continue
                    for model in obj.get_supported_models():
                        assert model.name in [
                            model.name for model in models
                        ], f"{model.name} not in models"

                        found_models.append(model.name)

    assert len(models) > 0
    # assert len(models) == len(found_models)


def test_all_supported_models_work_ok():
    models = AdapterFactory.get_supported_models()
    for model in models:
        assert AdapterFactory.get_adapter_by_path(model.get_path())
