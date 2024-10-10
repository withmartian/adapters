from adapters.adapter_factory import AdapterFactory
from adapters.provider_adapters.gemini_sdk_chat_provider_adapter import GeminiModel

N_PARAM = 2
MAX_TOKENS = 5
MODEL_PATHS = [
    model.get_path()
    for model in AdapterFactory.get_supported_models()
    if isinstance(model, GeminiModel) is False
]
MODEL_PATHS_ASYNC = [
    model.get_path()
    for model in AdapterFactory.get_supported_models()
    if model._test_async
]
