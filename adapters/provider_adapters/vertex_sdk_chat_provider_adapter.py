# from typing import Any, Dict, Literal, Optional
# from vertexai.generative_models import GenerativeModel
# from google.cloud.aiplatform_v1.services.prediction_service import (
#     PredictionServiceClient,
#     PredictionServiceAsyncClient,
# )
# from google.api_core.client_options import ClientOptions
# from openai import NOT_GIVEN, NotGiven
# from openai.types.chat import ChatCompletionMessage
# from openai.types.chat.chat_completion import Choice

# from adapters.abstract_adapters.api_key_adapter_mixin import ApiKeyAdapterMixin
# from adapters.abstract_adapters.sdk_chat_adapter import SDKChatAdapter
# from adapters.types import (
#     AdapterChatCompletion,
#     ContentTurn,
#     Conversation,
#     ConversationRole,
#     Cost,
#     Model,
#     ModelProperties,
#     Turn,
# )
# from adapters.utils.general_utils import get_dynamic_cost

# PROVIDER_NAME = "gemini"
# API_KEY_NAME = "GEMINI_API_KEY"
# BASE_PROPERTIES = ModelProperties(gdpr_compliant=True)


# class GeminiModel(Model):
#     _test_async: bool = False

#     vendor_name: str = PROVIDER_NAME
#     provider_name: str = PROVIDER_NAME
#     properties: ModelProperties = BASE_PROPERTIES

#     supports_repeating_roles: bool = True
#     supports_system: bool = True
#     supports_multiple_system: bool = True
#     supports_empty_content: bool = True
#     supports_tool_choice_required: bool = True
#     supports_last_assistant: bool = True
#     supports_first_assistant: bool = True


# MODELS = [
#     GeminiModel(
#         name="gemini-1.0-pro",
#         cost=Cost(prompt=0.5e-6, completion=1.5e-6),
#         context_length=30720,
#         completion_length=2048,
#     ),
#     GeminiModel(
#         name="gemini-1.5-pro",
#         cost=Cost(prompt=3.5e-6, completion=10.5e-6),
#         context_length=128000,
#         completion_length=8192,
#     ),
#     GeminiModel(
#         name="gemini-1.5-flash",
#         cost=Cost(prompt=0.35e-6, completion=0.70e-6),
#         context_length=128000,
#         completion_length=8192,
#     ),
# ]


# # TODO: max_tokens doesnt work
# class GeminiSDKChatProviderAdapter(
#     ProviderAdapterMixin,
#     ApiKeyAdapterMixin,
#     SDKChatAdapter,
# ):
#     @staticmethod
#     def get_supported_models():
#         return MODELS

#     @staticmethod
#     def get_provider_name() -> str:
#         return PROVIDER_NAME

#     def get_base_sdk_url(self) -> str:
#         return ""

#     @staticmethod
#     def get_api_key_name() -> str:
#         return API_KEY_NAME

#     _sync_client: PredictionServiceClient
#     _async_client: PredictionServiceAsyncClient

#     def __init__(
#         self,
#     ) -> None:
#         super().__init__()
#         self.set_api_key(self.get_api_key())
#         self.model: GenerativeModel | None = None

#     def get_model_name(self) -> str:
#         if self._current_model is None:
#             raise ValueError("Model not set")
#         return self._current_model.name

#     def get_async_client(self):
#         return self._async_client

#     def get_sync_client(self):
#         return self._sync_client

#     def adjust_temperature(self, temperature: float) -> float:
#         return temperature / 2

#     def set_api_key(self, api_key: str) -> None:
#         super().set_api_key(api_key)

#         self._sync_client = PredictionServiceClient(
#             client_options=ClientOptions(api_key=api_key), transport="rest"
#         )
#         self._async_client = PredictionServiceAsyncClient(
#             client_options=ClientOptions(api_key=api_key)
#         )

#     def extract_response(self, request: Any, response: Any) -> AdapterChatCompletion:
#         choices = [
#             Choice(
#                 message=ChatCompletionMessage(
#                     role=ConversationRole.assistant.value,
#                     content=response.text,
#                 ),
#                 finish_reason="stop",
#                 index=0,
#             )
#         ]

#         # Optimize token count calculation, use async for async and parallelize
#         assert self.model
#         prompt_tokens = self.model.count_tokens(
#             [_map_turn_content_to_str(turn) for turn in request.turns]
#         ).total_tokens
#         completion_tokens = self.model.count_tokens(response.text).total_tokens

#         dynamic_cost = get_dynamic_cost(self.get_model_name(), prompt_tokens)
#         cost = (
#             dynamic_cost.prompt * prompt_tokens
#             + dynamic_cost.completion * completion_tokens
#             + self.get_model().cost.request
#         )

#         return AdapterChatCompletion(
#             response=Turn(
#                 role=ConversationRole.assistant,
#                 content=choices[0]["message"]["content"],  # type: ignore
#             ),  # TODO: Refactor response
#             choices=choices,
#             cost=cost,
#             token_counts=Cost(
#                 prompt=prompt_tokens,
#                 completion=completion_tokens,
#             ),
#         )

#     async def extract_response_async(
#         self, request: Any, response: Any
#     ) -> AdapterChatCompletion:
#         model = GenerativeModel(model_name=self.get_model_name())
#         model._async_client = self.get_async_client()

#         choices = [
#             Choice(
#                 message=ChatCompletionMessage(
#                     role=ConversationRole.assistant.value,
#                     content=response.text,
#                 ),
#                 finish_reason="stop",
#                 index=0,
#             )
#         ]

#         prompt_tokens = await model.count_tokens_async(
#             [turn.content for turn in request.turns]
#         )
#         completion_tokens = await model.count_tokens_async(response.text)

#         dynamic_cost = get_dynamic_cost(self.get_model_name(), prompt_tokens)
#         cost = (
#             dynamic_cost.prompt * prompt_tokens
#             + dynamic_cost.completion * completion_tokens
#             + self.get_model().cost.request
#         )

#         return AdapterChatCompletion(
#             response=Turn(
#                 role=ConversationRole.assistant,
#                 content=choices[0]["message"]["content"],  # type: ignore
#             ),
#             choices=choices,
#             cost=cost,
#             token_counts=Cost(
#                 prompt=prompt_tokens.total_tokens,
#                 completion=completion_tokens.total_tokens,
#             ),
#         )

#     def extract_stream_response(self, request, response):
#         return response

#     def get_params(
#         self,
#         llm_input: Conversation,
#         **kwargs,
#     ) -> Dict[str, Any]:
#         params = super().get_params(llm_input, **kwargs)

#         last_message = params["messages"].pop()
#         last_content = last_message["content"] or " "

#         transformed_messages = []
#         for message in params["messages"]:
#             transformed_message = {
#                 "parts": [_map_content_to_str(message["content"], message["role"])],
#                 "role": _map_role(message["role"]),
#             }
#             transformed_messages.append(transformed_message)

#         if "max_tokens" in params:
#             params["max_output_tokens"] = params["max_tokens"]
#             del params["max_tokens"]

#         del params["messages"]

#         return {
#             "config": params,
#             "history": transformed_messages,
#             "prompt": _map_content_to_str(last_content, last_message["role"]),
#         }

#     async def execute_async(
#         self,
#         llm_input: Conversation,
#         stream: Optional[Literal[False]] | Literal[True] | NotGiven = NOT_GIVEN,
#         **kwargs,
#     ):
#         params = self.get_params(llm_input, **kwargs)

#         model = GenerativeModel(
#             model_name=self.get_model_name(),
#             generation_config=params["config"],
#         )
#         model._prediction_async_client_value = self.get_async_client()

#         convo = model.start_chat(history=params["history"])

#         result = await convo.send_message_async(params["prompt"])

#         return await self.extract_response_async(request=llm_input, response=result)

#     def execute_sync(
#         self,
#         llm_input: Conversation,
#         stream: Optional[Literal[False]] | Literal[True] | NotGiven = NOT_GIVEN,
#         **kwargs,
#     ):
#         params = self.get_params(llm_input, **kwargs)

#         self.model = GenerativeModel(
#             model_name=self.get_model_name(),
#             generation_config=params.get("config") or None,
#         )
#         self.model._prediction_client_value = self.get_sync_client()

#         chat = self.model.start_chat(history=params["history"])
#         result = chat.send_message(params["prompt"])
#         return self.extract_response(request=llm_input, response=result)


# def _map_content_to_str(
#     content: str | list[dict[str, Any]], role: str | ConversationRole
# ) -> str:
#     if not content or content in [" ", "\n", "\t"]:
#         return "."

#     # Join content if it is a list.
#     if isinstance(content, list):
#         return " ".join(content.get("text", "") for content in content)

#     if role != "system":
#         return content

#     # Simulate *system message*.
#     return f"*{content}*"


# def _map_role(role: str | ConversationRole) -> str:
#     match role:
#         case ConversationRole.user | ConversationRole.system | "user" | "system":
#             return "user"
#         case ConversationRole.assistant | "assistant":
#             return "model"
#         case _:
#             return "user"


# def _map_turn_content_to_str(turn: ContentTurn | Turn) -> str:
#     match turn:
#         case ContentTurn():
#             return " ".join(content.text for content in turn.content)  # type: ignore[union-attr]
#         case Turn():
#             return turn.content
