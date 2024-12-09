# from typing import Any, Callable, Dict

# from openai import AsyncAzureOpenAI, AzureOpenAI, OpenAI
# from openai.types.chat.chat_completion_chunk import (
#     ChatCompletionChunk,
#     Choice as ChoiceChunk,
#     ChoiceDelta,
# )

# from adapters.abstract_adapters.openai_sdk_chat_adapter import OpenAISDKChatAdapter
# from adapters.types import (
#     AdapterChatCompletionChunk,
#     Conversation,
#     ConversationRole,
#     Cost,
#     Model,
#     ModelProperties,
#     Provider,
#     Vendor,
# )


# class AzureModel(Model):
#     provider_name: str = Provider.azure.value

#     properties: ModelProperties = ModelProperties(gdpr_compliant=True)


# MODELS: list[Model] = [
#     AzureModel(
#         name="gpt-4o",
#         cost=Cost(prompt=5.0e-6, completion=15.0e-6),
#         context_length=128000,
#         completion_length=4096,
#         vendor_name=Vendor.openai.value,
#     ),
#     AzureModel(
#         name="gpt-4o-mini",
#         cost=Cost(prompt=0.15e-6, completion=0.6e-6),
#         context_length=128000,
#         completion_length=16385,
#         vendor_name=Vendor.openai.value,
#     ),
# ]


# class AzureSDKChatProviderAdapter(OpenAISDKChatAdapter):
#     @staticmethod
#     def get_supported_models() -> list[Model]:
#         return MODELS

#     @staticmethod
#     def get_api_key_name() -> str:
#         return "AZURE_API_KEY"

#     def get_base_sdk_url(self) -> str:
#         return "https://martiantest.openai.azure.com/"

#     def _call_sync(self) -> Callable[..., Any]:
#         return self._client_sync.chat.completions.create

#     def _call_async(self) -> Callable[..., Any]:
#         return self._client_async.chat.completions.create

#     def _create_client_sync(self, base_url: str, api_key: str) -> OpenAI:
#         return AzureOpenAI(
#             api_key=api_key,
#             azure_endpoint=base_url,
#             api_version="2024-06-01",
#         )

#     def _create_client_async(self, base_url: str, api_key: str) -> AsyncAzureOpenAI:
#         return AsyncAzureOpenAI(
#             api_key=api_key,
#             azure_endpoint=base_url,
#             api_version="2024-06-01",
#         )

#     def _get_params(self, llm_input: Conversation, **kwargs: Any) -> Dict[str, Any]:
#         params = super()._get_params(llm_input, **kwargs)

#         azure_tool_choice = kwargs.get("tool_choice")

#         if azure_tool_choice == "required":
#             azure_tool_choice = "auto"

#         return {
#             **params,
#             "tool_choice": azure_tool_choice,
#         }

#     def _extract_stream_response(
#         self, request: Any, response: ChatCompletionChunk, state: dict[str, Any]
#     ) -> AdapterChatCompletionChunk:
#         adapter_response = AdapterChatCompletionChunk.model_construct(
#             **response.model_dump(),
#         )

#         if len(adapter_response.choices) == 0:
#             adapter_response.choices = [
#                 ChoiceChunk(
#                     index=0,
#                     delta=ChoiceDelta(
#                         role=ConversationRole.assistant.value, content=""
#                     ),
#                 )
#             ]

#         return adapter_response
