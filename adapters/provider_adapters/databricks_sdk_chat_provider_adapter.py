import re
from typing import Any, Dict, Pattern

from adapters.abstract_adapters.openai_sdk_chat_adapter import OpenAISDKChatAdapter
from adapters.abstract_adapters.provider_adapter_mixin import ProviderAdapterMixin
from adapters.types import Conversation, ConversationRole, Cost, Model

PROVIDER_NAME = "databricks"
DATABRICKS_BASE_URL = (
    "https://adb-8736858266948228.8.azuredatabricks.net/serving-endpoints"
)
API_KEY_NAME = "DATABRICKS_API_KEY"
API_KEY_PATTERN = re.compile(r".*")


# Convert rate from DBU to USD
CONVERT_RATE = 0.07


class DatabricksModel(Model):
    supports_first_assistant: bool = False
    supports_multiple_system: bool = False
    supports_tools: bool = False
    provider_name: str = PROVIDER_NAME


MODELS = [
    DatabricksModel(
        name="databricks-dbrx-instruct",
        cost=Cost(prompt=10.714 * CONVERT_RATE, completion=32.143 * CONVERT_RATE),
        context_length=32000,
        vendor_name="databricks",
    ),
    DatabricksModel(
        name="databricks-meta-llama-3-1-405b-instruct",
        cost=Cost(prompt=142.857 * CONVERT_RATE, completion=428.571 * CONVERT_RATE),
        context_length=128000,
        vendor_name="databricks",
    ),
    DatabricksModel(
        name="databricks-meta-llama-3-1-70b-instruct",
        cost=Cost(prompt=14.286 * CONVERT_RATE, completion=42.857 * CONVERT_RATE),
        context_length=8000,
        vendor_name="databricks",
    ),
    DatabricksModel(
        name="databricks-mixtral-8x7b-instruct",
        cost=Cost(prompt=7.143 * CONVERT_RATE, completion=14.286 * CONVERT_RATE),
        context_length=32000,
        vendor_name="databricks",
    ),
    DatabricksModel(
        name="databricks-llama-2-70b-chat",
        cost=Cost(prompt=7.143 * CONVERT_RATE, completion=21.429 * CONVERT_RATE),
        context_length=4096,
        vendor_name="databricks",
    ),
]


class DatabricksSDKChatProviderAdapter(ProviderAdapterMixin, OpenAISDKChatAdapter):
    def get_base_sdk_url(self) -> str:
        return DATABRICKS_BASE_URL

    @staticmethod
    def get_supported_models():
        return MODELS

    @staticmethod
    def get_provider_name() -> str:
        return PROVIDER_NAME

    @staticmethod
    def get_api_key_name() -> str:
        return API_KEY_NAME

    @staticmethod
    def get_api_key_pattern() -> Pattern:
        return API_KEY_PATTERN

    def get_params(
        self,
        llm_input: Conversation,
        **kwargs,
    ) -> Dict[str, Any]:
        params = super().get_params(llm_input, **kwargs)

        messages = params["messages"]
        databricksTools = kwargs.get("tools")

        # Databricks only support system as a first optional message
        if messages and messages[0]["role"] == ConversationRole.system:
            system_message = messages[0]
            messages = [system_message] + [
                msg for msg in messages[1:] if msg["role"] != ConversationRole.system
            ]
        else:
            messages = [
                msg for msg in messages if msg["role"] != ConversationRole.system
            ]

        # Databricks only support ending messages with user or tool roles
        if messages and messages[-1]["role"] not in [
            ConversationRole.user,
            ConversationRole.tool,
        ]:
            messages = messages + [{"role": ConversationRole.user, "content": ""}]

        if databricksTools and not databricksTools[0]["function"].get("parameters"):
            databricksTools[0]["function"]["parameters"] = {
                "type": "object",
            }

        return {
            **params,
            "messages": messages,
            "tools": databricksTools,
            "max_tokens": (
                kwargs.get("max_tokens")
                if kwargs.get("max_tokens")
                else self.get_model().completion_length
            ),
        }
