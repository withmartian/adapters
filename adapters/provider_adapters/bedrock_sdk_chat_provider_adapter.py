import json
import re
from typing import Any, Pattern

import boto3  # type: ignore

from adapters.abstract_adapters.api_key_adapter_mixin import ApiKeyAdapterMixin
from adapters.abstract_adapters.provider_adapter_mixin import ProviderAdapterMixin
from adapters.abstract_adapters.sdk_chat_adapter import SDKChatAdapter
from adapters.types import (
    AdapterException,
    AdapterStreamResponse,
    Conversation,
    ConversationRole,
    Cost,
    Model,
    OpenAIChatAdapterResponse,
    Turn,
)

PROVIDER_NAME = "bedrock-runtime"
BASE_URL = "bedrock.us-east-1.amazonaws.com"
API_SECRET_KEY_NAME = "AWS_BEDROCK_API_KEY"
API_KEY_ID = "AWS_BEDROCK_API_KEY_ID"
API_KEY_PATTERN = re.compile(r".*")
ANTHROPIC_VERSION = "bedrock-2023-05-31"
REGION_NAME = "us-east-1"
MAX_TOKENS = 1000


class BedrockModel(Model):
    supports_streaming: bool = True
    supports_functions: bool = True
    supports_tools: bool = True
    supports_n: bool = True
    supports_json_output: bool = True
    supports_json_content: bool = True
    supports_first_assistant: bool = False
    supports_system: bool = False
    supports_empty_content: bool = False
    vendor_name: str = PROVIDER_NAME
    provider_name: str = PROVIDER_NAME


MODELS = [
    # BedrockModel(
    #     name="jamba-instruct",
    #     cost=Cost(prompt=0.125e-6, completion=0.375e-6),
    #     context_length=30720,
    #     completion_length=2048,
    #     modelId = ''
    # ),
    # BedrockModel(
    #     name="jurassic-2-ultra",
    #     cost=Cost(prompt=0.125e-6, completion=0.375e-6),
    #     context_length=30720,
    #     completion_length=2048,
    #     modelId = ''
    # ),
    # BedrockModel(
    #     name="jurassic-2-mid",
    #     cost=Cost(prompt=3.5e-6, completion=10.5e-6),
    #     context_length=128000,
    #     completion_length=8192,
    #     modelId = ''
    # ),
    # BedrockModel(
    #     name="claude-3.5-sonnet",
    #     cost=Cost(prompt=3.5e-6, completion=10.5e-6),
    #     context_length=128000,
    #     completion_length=8192,
    #     modelId = ''
    # ),
    # BedrockModel(
    #     name="claude-3-opus",
    #     cost=Cost(prompt=0.35e-6, completion=0.70e-6),
    #     context_length=128000,
    #     completion_length=8192,
    #     modelId = ''
    # ),
    # BedrockModel(
    #     name="claude-3-haiku",
    #     cost=Cost(prompt=0.35e-6, completion=0.70e-6),
    #     context_length=128000,
    #     completion_length=8192,
    #     modelId = ''
    # ),
    BedrockModel(
        name="claude-3-sonnet",
        cost=Cost(prompt=0.35e-6, completion=0.70e-6),
        context_length=128000,
        completion_length=8192,
        modelId="anthropic.claude-3-sonnet-20240229-v1:0",
    ),
    # BedrockModel(
    #     name="claude-2.1",
    #     cost=Cost(prompt=0.35e-6, completion=0.70e-6),
    #     context_length=128000,
    #     completion_length=8192,
    #     modelId = ''
    # ),
    # BedrockModel(
    #     name="claude-2.0",
    #     cost=Cost(prompt=0.35e-6, completion=0.70e-6),
    #     context_length=128000,
    #     completion_length=8192,
    #     modelId = ''
    # ),
    # BedrockModel(
    #     name="claude-instant",
    #     cost=Cost(prompt=0.35e-6, completion=0.70e-6),
    #     context_length=128000,
    #     completion_length=8192,
    #     modelId = ''
    # ),
]


FINISH_REASON_MAPPING = {
    "end_turn": "stop",
    "max_tokens": "length",
}


class BedrockSDKChatProviderAdapter(
    ProviderAdapterMixin, ApiKeyAdapterMixin, SDKChatAdapter
):
    @staticmethod
    def get_supported_models():
        return MODELS

    @staticmethod
    def get_provider_name() -> str:
        return PROVIDER_NAME

    def get_base_sdk_url(self) -> str:
        return BASE_URL

    @staticmethod
    def get_api_key_name() -> str:
        return API_SECRET_KEY_NAME

    def get_api_key_id(self) -> str:
        return API_KEY_ID

    def get_anthropic_version(self) -> str:
        return ANTHROPIC_VERSION

    def get_max_tokens(self) -> int:
        return MAX_TOKENS

    def get_region_name(self) -> str:
        return REGION_NAME

    @staticmethod
    def get_api_key_pattern() -> Pattern:
        return API_KEY_PATTERN

    _sync_client: boto3.client

    # TODO: boto3 async support
    _async_client: boto3.client

    def __init__(
        self,
    ):
        super().__init__()
        self._sync_client = boto3.client(
            self.get_provider_name(),
            region_name=self.get_region_name(),
            aws_access_key_id=self.get_api_key_id(),
            aws_secret_access_key=self.get_api_key_name(),
        )
        # TODO: boto3 async support

    def get_async_client(self):
        return self._async_client

    def get_sync_client(self):
        return self._sync_client

    def execute_sync(
        self,
        llm_input: Conversation,
        **kwargs,
    ):
        params = self.get_params(llm_input, **kwargs)
        messages = params["messages"]

        # Bedrocks does not support trailing whitespace
        if messages and messages[-1]["role"] == ConversationRole.assistant:
            messages[-1]["content"] = messages[-1]["content"].rstrip()

        body_raw = {
            "anthropic_version": self.get_anthropic_version(),
            "max_tokens": self.get_max_tokens(),
            "messages": messages,
        }
        body = json.dumps(body_raw)

        model_response = self.get_sync_client().invoke_model(
            modelId=self.get_model().modelId, body=body
        )

        response = json.loads(model_response["body"].read())

        if params.get("stream", False):

            def stream_response():
                try:
                    for chunk in response:
                        yield self.extract_stream_response(
                            request=llm_input, response=chunk
                        )
                except Exception as e:
                    raise AdapterException(f"Error in streaming response: {e}") from e
                finally:
                    response.close()

            return AdapterStreamResponse(response=stream_response())

        return self.extract_response(request=llm_input, response=response)

    def extract_response(
        self, request: Any, response: Any
    ) -> OpenAIChatAdapterResponse:
        choices = [
            {
                "message": {
                    "role": response["role"],
                    "content": choice["text"],
                },
                "finish_reason": FINISH_REASON_MAPPING.get(
                    response["stop_reason"], response["stop_reason"]
                ),
            }
            for choice in response["content"]
        ]
        prompt_tokens = response["usage"]["input_tokens"]
        completion_tokens = response["usage"]["output_tokens"]
        cost = (
            self.get_model().cost.prompt * prompt_tokens
            + self.get_model().cost.completion * completion_tokens
            + self.get_model().cost.request
        )

        return OpenAIChatAdapterResponse(
            response=Turn(
                role=ConversationRole.assistant,
                content=choices[0]["message"]["content"],  # type: ignore
            ),  # TODO: Refactor response
            choices=choices,
            cost=cost,
            token_counts=Cost(
                prompt=prompt_tokens,
                completion=completion_tokens,
            ),
        )

    def extract_stream_response(self, request, response):
        content = getattr(getattr(response, "delta", None), "text", "")

        if getattr(response, "type", None) == "message_stop":
            content = None

        chunk = json.dumps(
            {
                "choices": [
                    {
                        "delta": {
                            "role": ConversationRole.assistant,
                            "content": content,
                        },
                    }
                ]
            }
        )

        return f"data: {chunk}\n\n"
