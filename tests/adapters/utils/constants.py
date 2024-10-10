from adapters.adapter_factory import AdapterFactory

N_PARAM = 2

MAX_TOKENS = 5

MODEL_PATHS = [model.get_path() for model in AdapterFactory.get_supported_models()]

MODEL_PATHS_ASYNC = [
    model.get_path()
    for model in AdapterFactory.get_supported_models()
    if model._test_async
]
