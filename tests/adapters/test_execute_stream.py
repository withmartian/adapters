# import pytest

# from adapters.adapter_factory import AdapterFactory
# from adapters.types import ConversationRole
# from tests.adapters.utils.contants import MODEL_PATHS
# from tests.utils import (
#     SIMPLE_CONVERSATION_USER_ONLY,
#     get_choices_from_vcr,
# )

# @pytest.mark.parametrize("model_name", MODEL_NAMES)
# @pytest.mark.vcr
# def test_sync_execute_streaming(vcr, model_name):
#     adapter = AdapterFactory.get_adapter_by_path(model_name)

#     assert adapter is not None


#     if adapter.get_model().supports_streaming is False:
#         return

#     adapter_response = adapter.execute_sync(
#         adapter.convert_to_input(SIMPLE_CONVERSATION_USER_ONLY),
#         stream=True,
#     )

#     chunks = [
#         json.loads(data_chunk[6:].strip()) for data_chunk in adapter_response.response
#     ]

#     response = "".join(
#         [
#             chunk["choices"][0]["delta"]["content"]
#             for chunk in chunks
#             if chunk["choices"][0]["delta"]["content"]
#         ]
#     )
#     assert len(response) > 0


# @pytest.mark.parametrize("model_name", MODEL_NAMES)
# @pytest.mark.vcr
# async def test_async_execute_streaming(model_name):
#     adapter = AdapterFactory.get_adapter_by_path(model_name)

#     assert adapter is not None


#     if adapter.get_model().supports_streaming is False:
#         return

#     adapter_response = await adapter.execute_async(
#         adapter.convert_to_input(SIMPLE_CONVERSATION_USER_ONLY),
#         stream=True,
#     )

#     chunks = [
#         json.loads(data_chunk[6:].strip())
#         async for data_chunk in adapter_response.response
#     ]

#     response = "".join(
#         [
#             chunk["choices"][0]["delta"]["content"]
#             for chunk in chunks
#             if chunk["choices"][0]["delta"]["content"]
#         ]
#     )
#     assert len(response) > 0


# # @pytest.mark.parametrize("model_name", ANTHROPIC_CHAT_MODELS)
# # @pytest.mark.vcr
# # async def test_async_execute_streaming(
# #     vcr, model_name
# # ):
# #     adapter = AdapterFactory.get_adapter_by_path(model_name)

# #     if not adapter.get_model().supports_streaming:
# #         return

# #     adapter_response = await adapter.execute_async(
# #         adapter.convert_to_input(SIMPLE_CONVERSATION_USER_ONLY),
# #         stream=True,
# #     )

# #     chunks = [
# #         json.loads(data_chunk[6:].strip())
# #         async for data_chunk in adapter_response.response
# #     ]

# #     response = "".join(
# #         [
# #             chunk["choices"][0]["delta"]["content"]
# #             for chunk in chunks
# #             if chunk["choices"][0]["delta"]["content"]
# #         ]
# #     )

# #     assert len(response) > 0


# # @pytest.mark.parametrize("model_name", ANTHROPIC_CHAT_MODELS)
# # @pytest.mark.vcr
# # def test_sync_execute_streaming(vcr, model_name):
# #     adapter = AdapterFactory.get_adapter_by_path(model_name)

# #     if not adapter.get_model().supports_streaming:
# #         return

# #     adapter_response = adapter.execute_sync(
# #         adapter.convert_to_input(SIMPLE_CONVERSATION_USER_ONLY),
# #         stream=True,
# #     )

# #     chunks = [
# #         json.loads(data_chunk[6:].strip()) for data_chunk in adapter_response.response
# #     ]

# #     response = "".join(
# #         [
# #             chunk["choices"][0]["delta"]["content"]
# #             for chunk in chunks
# #             if chunk["choices"][0]["delta"]["content"]
# #         ]
# #     )

# #     assert len(response) > 0
