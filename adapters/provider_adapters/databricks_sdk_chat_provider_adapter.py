from typing import Any, Dict

from adapters.abstract_adapters.openai_sdk_chat_adapter import OpenAISDKChatAdapter
from adapters.types import Conversation, ConversationRole, Cost, Model, ModelProperties

PROVIDER_NAME = "databricks"
DATABRICKS_BASE_URL = (
    "https://adb-8736858266948228.8.azuredatabricks.net/serving-endpoints"
)
API_KEY_NAME = "DATABRICKS_API_KEY"
BASE_PROPERTIES = ModelProperties(open_source=True, gdpr_compliant=True)

DBU_USD_RATE = 0.07


class DatabricksModel(Model):
    provider_name: str = PROVIDER_NAME
    properties: ModelProperties = BASE_PROPERTIES

    supports_repeating_roles: bool = True
    supports_system: bool = True
    supports_empty_content: bool = True
    supports_tool_choice_required: bool = True
    supports_last_assistant: bool = True
    supports_streaming: bool = True
    supports_temperature: bool = True


MODELS = [
    DatabricksModel(
        name="databricks-meta-llama-3-1-70b-instruct",
        cost=Cost(prompt=14.286 * DBU_USD_RATE, completion=42.857 * DBU_USD_RATE),
        context_length=8000,
        vendor_name="meta-llama",
    ),
    DatabricksModel(
        name="databricks-meta-llama-3-1-405b-instruct",
        cost=Cost(prompt=71.429 * DBU_USD_RATE, completion=214.286 * DBU_USD_RATE),
        context_length=128000,
        vendor_name="meta-llama",
    ),
    DatabricksModel(
        name="databricks-mixtral-8x7b-instruct",
        cost=Cost(prompt=7.143 * DBU_USD_RATE, completion=14.286 * DBU_USD_RATE),
        context_length=32000,
        vendor_name="mistralai",
    ),
    DatabricksModel(
        name="databricks-mixtral-8x7b-instruct",
        cost=Cost(prompt=7.143 * DBU_USD_RATE, completion=14.286 * DBU_USD_RATE),
        context_length=32000,
        vendor_name="databricks",
    ),
    DatabricksModel(
        name="databricks-dbrx-instruct",
        cost=Cost(prompt=10.714 * DBU_USD_RATE, completion=32.143 * DBU_USD_RATE),
        context_length=32000,
        completion_length=4000,
        vendor_name="databricks",
    ),
]


class DatabricksSDKChatProviderAdapter(OpenAISDKChatAdapter):
    @staticmethod
    def get_supported_models():
        return MODELS

    @staticmethod
    def get_provider_name() -> str:
        return PROVIDER_NAME

    @staticmethod
    def get_api_key_name() -> str:
        return API_KEY_NAME

    def get_base_sdk_url(self) -> str:
        return DATABRICKS_BASE_URL

    def _get_params(
        self,
        llm_input: Conversation,
        **kwargs,
    ) -> Dict[str, Any]:
        params = super()._get_params(llm_input, **kwargs)

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
