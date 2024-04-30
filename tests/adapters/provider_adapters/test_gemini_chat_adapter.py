# import json
# from typing import List

# import pytest

# from adapters import AdapterFactory
# from adapters.provider_adapters.gemini_sdk_chat_provider_adapter import (
#     GeminiSDKChatProviderAdapter,
# )
# from adapters.types import ConversationRole, Model
# from tests.utils import SIMPLE_CONVERSATION_USER_ONLY

# MODELS: List[Model] = [
#     *GeminiSDKChatProviderAdapter.get_supported_models(),
# ]

# MODEL_NAMES = [f"{model.name}" for model in MODELS]


# @pytest.mark.parametrize("model_name", MODEL_NAMES)
# @pytest.mark.vcr
# def test_sync_execute_on_gemini_models_ok(vcr, model_name):
#     adapter = AdapterFactory.get_adapter(model_name)
#     adapter_response = adapter.execute_sync(
#         adapter.convert_to_input(SIMPLE_CONVERSATION_USER_ONLY)
#     )

#     cassette_response = json.loads(vcr.responses[0]["body"]["string"])

#     assert (
#         adapter_response.response.content
#         == cassette_response["candidates"][0]["content"]["parts"][0]["text"]
#     )
#     assert adapter_response.response.role == ConversationRole.assistant
#     assert adapter_response.cost > 0


# @pytest.mark.parametrize("model_name", MODEL_NAMES)
# @pytest.mark.vcr
# async def test_async_execute_on_gemini_models_ok(vcr, model_name):
#     adapter = AdapterFactory.get_adapter(model_name)
#     adapter_response = await adapter.execute_async(
#         adapter.convert_to_input(SIMPLE_CONVERSATION_USER_ONLY)
#     )

#     cassette_response = json.loads(vcr.responses[0]["body"]["string"])

#     assert (
#         adapter_response.response.content
#         == cassette_response["candidates"][0]["content"]["parts"][0]["text"]
#     )
#     assert adapter_response.response.role == ConversationRole.assistant
#     assert adapter_response.cost > 0
