import re
from typing import Pattern

from adapters.abstract_adapters.openai_sdk_chat_adapter import OpenAISDKChatAdapter
from adapters.abstract_adapters.provider_adapter_mixin import ProviderAdapterMixin
from adapters.types import Cost, Model

PROVIDER_NAME = "together"
BASE_URL = "https://api.together.xyz"
API_KEY_NAME = "TOGETHER_API_KEY"
API_KEY_PATTERN = re.compile(r".*")


class TogetherModel(Model):
    supports_streaming: bool = True
    supports_json_content: bool = True
    provider_name: str = PROVIDER_NAME

    def _get_api_path(self) -> str:
        return f"{self.vendor_name}/{self.name}"


MODELS = [
    TogetherModel(
        name="Yi-34B-Chat",
        cost=Cost(prompt=0.8e-6, completion=0.8e-6),
        context_length=4096,
        vendor_name="zero-one-ai",
    ),
    TogetherModel(
        name="OLMo-7B-Instruct",
        cost=Cost(prompt=0.2e-6, completion=0.2e-6),
        context_length=2048,
        vendor_name="allenai",
    ),
    TogetherModel(
        name="OLMo-7B-Twin-2T",
        cost=Cost(prompt=0.2e-6, completion=0.2e-6),
        context_length=2048,
        vendor_name="allenai",
    ),
    TogetherModel(
        name="OLMo-7B",
        cost=Cost(prompt=0.2e-6, completion=0.2e-6),
        context_length=2048,
        vendor_name="allenai",
    ),
    TogetherModel(
        name="chronos-hermes-13b",
        cost=Cost(prompt=0.3e-6, completion=0.3e-6),
        context_length=2048,
        vendor_name="Austism",
    ),
    TogetherModel(
        name="dolphin-2.5-mixtral-8x7b",
        cost=Cost(prompt=0.6e-6, completion=0.6e-6),
        context_length=32768,
        vendor_name="cognitivecomputations",
    ),
    TogetherModel(
        name="dbrx-instruct",
        cost=Cost(prompt=1.2e-6, completion=1.2e-6),
        context_length=32768,
        vendor_name="databricks",
    ),
    TogetherModel(
        name="deepseek-coder-33b-instruct",
        cost=Cost(prompt=0.8e-6, completion=0.8e-6),
        context_length=16384,
        vendor_name="deepseek-ai",
    ),
    TogetherModel(
        name="deepseek-llm-67b-chat",
        cost=Cost(prompt=0.9e-6, completion=0.9e-6),
        context_length=4096,
        vendor_name="deepseek-ai",
    ),
    TogetherModel(
        name="Platypus2-70B-instruct",
        cost=Cost(prompt=0.9e-6, completion=0.9e-6),
        context_length=4096,
        vendor_name="garage-bAInd",
    ),
    TogetherModel(
        name="gemma-2b-it",
        cost=Cost(prompt=0.1e-6, completion=0.1e-6),
        context_length=8192,
        vendor_name="google",
    ),
    TogetherModel(
        name="gemma-7b-it",
        cost=Cost(prompt=0.2e-6, completion=0.2e-6),
        context_length=8192,
        vendor_name="google",
    ),
    TogetherModel(
        name="MythoMax-L2-13b",
        cost=Cost(prompt=0.3e-6, completion=0.3e-6),
        context_length=4096,
        vendor_name="Gryphe",
    ),
    TogetherModel(
        name="vicuna-13b-v1.5",
        cost=Cost(prompt=0.3e-6, completion=0.3e-6),
        context_length=4096,
        vendor_name="lmsys",
    ),
    TogetherModel(
        name="CodeLlama-7b-Instruct-hf",
        cost=Cost(prompt=0.2e-6, completion=0.2e-6),
        context_length=16384,
        vendor_name="codellama",
    ),
    TogetherModel(
        name="CodeLlama-13b-Instruct-hf",
        cost=Cost(prompt=0.22e-6, completion=0.22e-6),
        context_length=16384,
        vendor_name="codellama",
    ),
    TogetherModel(
        name="CodeLlama-34b-Instruct-hf",
        cost=Cost(prompt=0.78e-6, completion=0.78e-6),
        context_length=16384,
        vendor_name="codellama",
    ),
    # TogetherModel(
    #     name="CodeLlama-70b-Instruct-hf",
    #     cost=Cost(prompt=0.9e-6, completion=0.9e-6),
    #     context_length=4096,
    #     vendor_name="codellama",
    # ),
    TogetherModel(
        name="Llama-2-7b-chat-hf",
        cost=Cost(prompt=0.2e-6, completion=0.2e-6),
        context_length=4096,
        vendor_name="meta-llama",
    ),
    TogetherModel(
        name="Llama-2-13b-chat-hf",
        cost=Cost(prompt=0.22e-6, completion=0.22e-6),
        context_length=4096,
        vendor_name="meta-llama",
    ),
    TogetherModel(
        name="Llama-2-70b-chat-hf",
        cost=Cost(prompt=0.9e-6, completion=0.9e-6),
        context_length=4096,
        vendor_name="meta-llama",
    ),
    TogetherModel(
        name="Llama-3-8b-chat-hf",
        cost=Cost(prompt=0.2e-6, completion=0.2e-6),
        context_length=8192,
        vendor_name="meta-llama",
        supports_json_content=False,
    ),
    TogetherModel(
        name="Llama-3-70b-chat-hf",
        cost=Cost(prompt=0.9e-6, completion=0.9e-6),
        context_length=8000,
        vendor_name="meta-llama",
        supports_json_content=False,
    ),
    TogetherModel(
        name="Mistral-7B-Instruct-v0.1",
        cost=Cost(prompt=0.2e-6, completion=0.2e-6),
        context_length=4096,
        vendor_name="mistralai",
    ),
    # TogetherModel(
    #     name="Mixtral-8x7B-Instruct-v0.1",
    #     cost=Cost(prompt=0.6e-6, completion=0.6e-6),
    #     context_length=32768,
    #     vendor_name="mistralai",
    #     supports_json_output=True,
    # ),
    TogetherModel(
        name="Mixtral-8x22B-Instruct-v0.1",
        cost=Cost(prompt=1.2e-6, completion=1.2e-6),
        context_length=65536,
        vendor_name="mistralai",
    ),
    # TogetherModel(
    #     name="Mistral-7B-Instruct-v0.2",
    #     cost=Cost(prompt=0.2e-6, completion=0.2e-6),
    #     context_length=32768,
    #     vendor_name="mistralai",
    #     supports_multiple_system=False,
    #     supports_repeating_roles=False,
    # ),
    TogetherModel(
        name="Nous-Capybara-7B-V1p9",
        cost=Cost(prompt=0.2e-6, completion=0.2e-6),
        context_length=8192,
        vendor_name="NousResearch",
    ),
    TogetherModel(
        name="Nous-Hermes-2-Mixtral-8x7B-DPO",
        cost=Cost(prompt=0.6e-6, completion=0.6e-6),
        context_length=32768,
        vendor_name="NousResearch",
        supports_multiple_system=False,
    ),
    TogetherModel(
        name="Nous-Hermes-2-Mixtral-8x7B-SFT",
        cost=Cost(prompt=0.6e-6, completion=0.6e-6),
        context_length=32768,
        vendor_name="NousResearch",
    ),
    TogetherModel(
        name="Nous-Hermes-llama-2-7b",
        cost=Cost(prompt=0.2e-6, completion=0.2e-6),
        context_length=4096,
        vendor_name="NousResearch",
    ),
    TogetherModel(
        name="Nous-Hermes-Llama2-13b",
        cost=Cost(prompt=0.3e-6, completion=0.3e-6),
        context_length=4096,
        vendor_name="NousResearch",
    ),
    TogetherModel(
        name="Nous-Hermes-2-Yi-34B",
        cost=Cost(prompt=0.8e-6, completion=0.8e-6),
        context_length=4096,
        vendor_name="NousResearch",
    ),
    TogetherModel(
        name="openchat-3.5-1210",
        cost=Cost(prompt=0.2e-6, completion=0.2e-6),
        context_length=8192,
        vendor_name="openchat",
    ),
    TogetherModel(
        name="Mistral-7B-OpenOrca",
        cost=Cost(prompt=0.2e-6, completion=0.2e-6),
        context_length=8192,
        vendor_name="Open-Orca",
    ),
    TogetherModel(
        name="Qwen1.5-0.5B-Chat",
        cost=Cost(prompt=0.1e-6, completion=0.1e-6),
        context_length=32768,
        vendor_name="Qwen",
    ),
    TogetherModel(
        name="Qwen1.5-1.8B-Chat",
        cost=Cost(prompt=0.1e-6, completion=0.1e-6),
        context_length=32768,
        vendor_name="Qwen",
    ),
    TogetherModel(
        name="Qwen1.5-4B-Chat",
        cost=Cost(prompt=0.1e-6, completion=0.1e-6),
        context_length=32768,
        vendor_name="Qwen",
    ),
    TogetherModel(
        name="Qwen1.5-7B-Chat",
        cost=Cost(prompt=0.2e-6, completion=0.2e-6),
        context_length=32768,
        vendor_name="Qwen",
    ),
    TogetherModel(
        name="Qwen1.5-14B-Chat",
        cost=Cost(prompt=0.3e-6, completion=0.3e-6),
        context_length=32768,
        vendor_name="Qwen",
    ),
    TogetherModel(
        name="Qwen1.5-32B-Chat",
        cost=Cost(prompt=0.8e-6, completion=0.8e-6),
        context_length=32768,
        vendor_name="Qwen",
    ),
    TogetherModel(
        name="Qwen1.5-72B-Chat",
        cost=Cost(prompt=0.9e-6, completion=0.9e-6),
        context_length=4096,
        vendor_name="Qwen",
    ),
    # TogetherModel(
    #     name="Snorkel-Mistral-PairRM-DPO",
    #     cost=Cost(prompt=0.2e-6, completion=0.2e-6),
    #     context_length=32768,
    #     vendor_name="snorkelai",
    #     supports_first_assistant=False,
    # ),
    TogetherModel(
        name="alpaca-7b",
        cost=Cost(prompt=0.2e-6, completion=0.2e-6),
        context_length=2048,
        vendor_name="togethercomputer",
    ),
    TogetherModel(
        name="OpenHermes-2-Mistral-7B",
        cost=Cost(prompt=0.2e-6, completion=0.2e-6),
        context_length=8192,
        vendor_name="teknium",
    ),
    TogetherModel(
        name="OpenHermes-2p5-Mistral-7B",
        cost=Cost(prompt=0.2e-6, completion=0.2e-6),
        context_length=8192,
        vendor_name="teknium",
    ),
    TogetherModel(
        name="Llama-2-7B-32K-Instruct",
        cost=Cost(prompt=0.2e-6, completion=0.2e-6),
        context_length=32768,
        vendor_name="togethercomputer",
    ),
    TogetherModel(
        name="RedPajama-INCITE-Chat-3B-v1",
        cost=Cost(prompt=0.1e-6, completion=0.1e-6),
        context_length=2048,
        vendor_name="togethercomputer",
    ),
    TogetherModel(
        name="RedPajama-INCITE-7B-Chat",
        cost=Cost(prompt=0.2e-6, completion=0.2e-6),
        context_length=2048,
        vendor_name="togethercomputer",
    ),
    TogetherModel(
        name="StripedHyena-Nous-7B",
        cost=Cost(prompt=0.2e-6, completion=0.2e-6),
        context_length=32768,
        vendor_name="togethercomputer",
        supports_empty_content=False,
    ),
    TogetherModel(
        name="ReMM-SLERP-L2-13B",
        cost=Cost(prompt=0.3e-6, completion=0.3e-6),
        context_length=4096,
        vendor_name="Undi95",
    ),
    TogetherModel(
        name="Toppy-M-7B",
        cost=Cost(prompt=0.2e-6, completion=0.2e-6),
        context_length=4096,
        vendor_name="Undi95",
    ),
    TogetherModel(
        name="WizardLM-13B-V1.2",
        cost=Cost(prompt=0.3e-6, completion=0.3e-6),
        context_length=4096,
        vendor_name="WizardLM",
    ),
    # TogetherModel(
    #     name="SOLAR-10.7B-Instruct-v1.0",
    #     cost=Cost(prompt=0.3e-6, completion=0.3e-6),
    #     context_length=4096,
    #     vendor_name="upstage",
    # ),
]


class TogetherSDKChatProviderAdapter(ProviderAdapterMixin, OpenAISDKChatAdapter):
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
        return API_KEY_NAME

    @staticmethod
    def get_api_key_pattern() -> Pattern:
        return API_KEY_PATTERN

    def adjust_temperature(self, temperature: float) -> float:
        return temperature / 2
