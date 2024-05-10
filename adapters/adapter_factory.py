import inspect
import os
import sys
from typing import Any

from adapters.abstract_adapters import BaseAdapter
from adapters.abstract_adapters.provider_adapter_mixin import ProviderAdapterMixin
from adapters.concrete_adapters import *
from adapters.provider_adapters.anthropic_sdk_chat_provider_adapter import (
    AnthropicSDKChatProviderAdapter,
)
from adapters.provider_adapters.gemini_sdk_chat_provider_adapter import (
    GeminiSDKChatProviderAdapter,
)
from adapters.provider_adapters.openai_sdk_chat_provider_adapter import (
    OpenAISDKChatProviderAdapter,
)
from adapters.provider_adapters.together_sdk_chat_provider_adapter import (
    TogetherSDKChatProviderAdapter,
)
from adapters.types import Model


class AdapterFactory:
    @staticmethod
    def _create_adapter_registry() -> dict[str, type[BaseAdapter]]:
        adapters_classes: dict[str, type[BaseAdapter[Any, Any, Any, Any]]] = {}

        for _, obj in inspect.getmembers(sys.modules["adapters.provider_adapters"]):
            if (
                inspect.isclass(obj)
                and issubclass(obj, ProviderAdapterMixin)
                and issubclass(obj, BaseAdapter)
            ):
                for model in obj.get_supported_models():
                    adapters_classes[model.get_path()] = obj

        for model in OpenAISDKChatProviderAdapter.get_supported_models():
            adapters_classes[model.name] = OpenAISDKChatProviderAdapter  # type: ignore

        for model in AnthropicSDKChatProviderAdapter.get_supported_models():
            adapters_classes[model.name] = AnthropicSDKChatProviderAdapter  # type: ignore

        for model in TogetherSDKChatProviderAdapter.get_supported_models():
            adapters_classes[model.name] = TogetherSDKChatProviderAdapter  # type: ignore

        for model in GeminiSDKChatProviderAdapter.get_supported_models():
            adapters_classes[model.name] = GeminiSDKChatProviderAdapter  # type: ignore

        for _, obj in inspect.getmembers(sys.modules["adapters.concrete_adapters"]):
            if inspect.isclass(obj) and issubclass(obj, BaseAdapter):
                adapters_classes[obj().get_model().get_path()] = obj

        return adapters_classes

    @staticmethod
    def _create_model_registry() -> dict[str, Model]:
        models: dict[str, Model] = {}

        for _, obj in inspect.getmembers(sys.modules["adapters.provider_adapters"]):
            if (
                inspect.isclass(obj)
                and issubclass(obj, ProviderAdapterMixin)
                and issubclass(obj, BaseAdapter)
            ):
                for model in obj.get_supported_models():
                    models[model.get_path()] = model

        for model in OpenAISDKChatProviderAdapter.get_supported_models():
            models[model.name] = model

        for model in AnthropicSDKChatProviderAdapter.get_supported_models():
            models[model.name] = model

        for model in TogetherSDKChatProviderAdapter.get_supported_models():
            models[model.name] = model

        for model in GeminiSDKChatProviderAdapter.get_supported_models():
            models[model.name] = model

        for _, obj in inspect.getmembers(sys.modules["adapters.concrete_adapters"]):
            if inspect.isclass(obj) and issubclass(obj, BaseAdapter):
                model = obj().get_model()
                models[model.get_path()] = model

        return models

    @staticmethod
    def _create_model_list() -> list[Model]:
        models: list[Model] = []

        for _, obj in inspect.getmembers(sys.modules["adapters.provider_adapters"]):
            if (
                inspect.isclass(obj)
                and issubclass(obj, ProviderAdapterMixin)
                and issubclass(obj, BaseAdapter)
            ):
                for model in obj.get_supported_models():
                    models.append(model)

        return models

    _adapter_registry = _create_adapter_registry()

    # TODO: Doesn't work with concrete adapters
    _model_registry = _create_model_registry()
    # TODO: Doesn't work with concrete adapters
    _model_list = _create_model_list()

    @staticmethod
    def get_adapter_by_path(model_path: str) -> BaseAdapter | None:
        adapter_class = AdapterFactory._adapter_registry.get(model_path)
        model = AdapterFactory._model_registry.get(model_path)

        if adapter_class is None or model is None:
            return None

        adapter = adapter_class()

        if isinstance(adapter, ProviderAdapterMixin):
            adapter._set_current_model(model)

        return adapter

    @staticmethod
    def get_adapter(model: Model) -> BaseAdapter | None:
        adapter_class = AdapterFactory._adapter_registry.get(model.get_path())

        if adapter_class is None:
            return None

        adapter = adapter_class()

        if isinstance(adapter, ProviderAdapterMixin):
            adapter._set_current_model(model)

        return adapter

    @staticmethod
    def get_model_by_path(model_path: str) -> Model | None:
        return AdapterFactory._model_registry.get(model_path)

    @staticmethod
    def get_supported_models(
        supports_streaming: bool = False,
        supports_vision: bool = False,
        supports_functions: bool = False,
        supports_tools: bool = False,
        supports_n: bool = False,
        supports_json_output: bool = False,
        supports_json_content: bool = False,
    ) -> list[Model]:
        disabled_models = os.getenv("ADAPTER_DISABLED_MODELS", "").split(",")

        return [
            model
            for model in AdapterFactory._model_list
            if (not supports_streaming or model.supports_streaming)
            and (not supports_vision or model.supports_vision)
            and (not supports_functions or model.supports_functions)
            and (not supports_tools or model.supports_tools)
            and (not supports_n or model.supports_n)
            and (not supports_json_output or model.supports_json_output)
            and (not supports_json_content or model.supports_json_content)
            and model.name not in disabled_models
        ]
