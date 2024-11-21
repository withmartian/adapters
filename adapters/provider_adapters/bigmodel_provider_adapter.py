from typing import Any, Dict, Literal, Optional, overload
from openai import NOT_GIVEN, NotGiven
import asyncio
from adapters.abstract_adapters.openai_sdk_chat_adapter import OpenAISDKChatAdapter
from adapters.types import (
    Cost,
    Model,
    ModelProperties,
    Provider,
    Vendor,
    ConversationRole,
    Conversation,
    AdapterChatCompletion,
    AdapterStreamAsyncChatCompletion,
)
from openai import RateLimitError, BadRequestError


class BigModelModel(Model):
    provider_name: str = Provider.bigmodel.value
    vendor_name: str = Vendor.bigmodel.value

    properties: ModelProperties = ModelProperties(
        gdpr_compliant=False,
    )


MODELS: list[Model] = [
    BigModelModel(
        name="glm-4-plus",
        cost=Cost(prompt=0.05, completion=0.05),  # ¥0.05 / 1k tokens
        context_length=128000,
        completion_length=4096,
        supports_vision=False,
        supports_tools=False,
        supports_tool_choice=False,
        supports_tool_choice_required=False,
        supports_n=False,
    ),
    BigModelModel(
        name="glm-4-0520",
        cost=Cost(prompt=0.1, completion=0.1),  # ¥0.1 / 1k tokens
        context_length=128000,
        completion_length=4096,
        supports_vision=False,
        supports_tools=False,
        supports_tool_choice=False,
        supports_tool_choice_required=False,
        supports_n=False,
    ),
    BigModelModel(
        name="glm-4-airx",
        cost=Cost(prompt=0.01, completion=0.01),  # ¥0.01 / 1k tokens
        context_length=8000,
        completion_length=4096,
        supports_vision=False,
        supports_tools=False,
        supports_tool_choice=False,
        supports_tool_choice_required=False,
        supports_n=False,
    ),
    BigModelModel(
        name="glm-4-air",
        cost=Cost(prompt=0.001, completion=0.001),  # ¥0.001 / 1k tokens
        context_length=128000,
        completion_length=4096,
        supports_vision=False,
        supports_tools=False,
        supports_tool_choice=False,
        supports_tool_choice_required=False,
        supports_n=False,
    ),
    BigModelModel(
        name="glm-4-long",
        cost=Cost(prompt=0.001, completion=0.001),  # ¥0.001 / 1k tokens
        context_length=1000000,  # 1M
        completion_length=4096,
        supports_vision=False,
        supports_tools=False,
        supports_tool_choice=False,
        supports_tool_choice_required=False,
        supports_n=False,
    ),
    BigModelModel(
        name="glm-4-flashx",
        cost=Cost(prompt=0.0001, completion=0.0001),  # ¥0.0001 / 1k tokens
        context_length=128000,
        completion_length=4096,
        supports_vision=False,
        supports_tools=False,
        supports_tool_choice=False,
        supports_tool_choice_required=False,
        supports_n=False,
    ),
    BigModelModel(
        name="glm-4-flash",
        cost=Cost(prompt=0.0, completion=0.0),  # Free
        context_length=128000,
        completion_length=4096,
        supports_vision=False,
        supports_tools=False,
        supports_tool_choice=False,
        supports_tool_choice_required=False,
        supports_n=False,
    ),
    BigModelModel(
        name="glm-4v",
        cost=Cost(prompt=0.05, completion=0.05),  # ¥0.05 / 1k tokens
        context_length=6000,
        completion_length=4096,
        supports_vision=True,
        supports_tools=False,
        supports_tool_choice=False,
        supports_tool_choice_required=False,
        supports_n=False,
    ),
    BigModelModel(
        name="glm-4v-plus",
        cost=Cost(prompt=0.01, completion=0.01),  # ¥0.01 / 1k tokens
        context_length=6000,
        completion_length=4096,
        supports_vision=True,
        supports_tools=False,
        supports_tool_choice=False,
        supports_tool_choice_required=False,
        supports_n=False,
    ),
]


class BigModelSDKChatProviderAdapter(OpenAISDKChatAdapter):
    MAX_RETRIES = 3
    RETRY_DELAY = 1

    @staticmethod
    def get_supported_models() -> list[Model]:
        return MODELS

    @staticmethod
    def get_api_key_name() -> str:
        return "BIGMODEL_API_KEY"

    def get_base_sdk_url(self) -> str:
        return "https://open.bigmodel.cn/api/paas/v4"

    def _prepare_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        # Modify the request before sending
        if "messages" in request:
            messages = request["messages"]

            # Filter out empty messages
            messages = [
                msg for msg in messages if msg.get("content") and msg["content"].strip()
            ]

            # Ensure there's at least one user message
            if not any(msg["role"] == ConversationRole.user for msg in messages):
                messages.append({"role": ConversationRole.user, "content": "Continue"})

            request["messages"] = messages

        return request

    @overload
    async def execute_async(
        self,
        llm_input: Conversation,
        stream: Optional[Literal[False]] | NotGiven = ...,
        **kwargs: Any,
    ) -> AdapterChatCompletion: ...

    @overload
    async def execute_async(
        self,
        llm_input: Conversation,
        stream: Literal[True],
        **kwargs: Any,
    ) -> AdapterStreamAsyncChatCompletion: ...

    async def execute_async(
        self,
        llm_input: Conversation,
        stream: Optional[Literal[False]] | Literal[True] | NotGiven = NOT_GIVEN,
        **kwargs: Any,
    ) -> AdapterChatCompletion | AdapterStreamAsyncChatCompletion:
        retries = 0
        while retries < self.MAX_RETRIES:
            try:
                kwargs = self._prepare_request(kwargs)
                return await super().execute_async(llm_input, stream=stream, **kwargs)
            except RateLimitError:
                if retries == self.MAX_RETRIES - 1:
                    raise
                retries += 1
                await asyncio.sleep(self.RETRY_DELAY * (2**retries))
            except BadRequestError as e:
                if "未正常接收到prompt参数" in str(e) or "messages 参数非法" in str(e):
                    kwargs["messages"] = [
                        {"role": ConversationRole.user, "content": "Hello"}
                    ]
                    return await super().execute_async(
                        llm_input, stream=stream, **kwargs
                    )
                raise

        return await super().execute_async(llm_input, stream=stream, **kwargs)
