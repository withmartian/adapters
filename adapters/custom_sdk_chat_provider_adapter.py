from adapters.abstract_adapters.openai_sdk_chat_adapter import OpenAISDKChatAdapter


class CustomOpenAISDKChatProviderAdapter(OpenAISDKChatAdapter):
    @staticmethod
    def get_supported_models():
        return []
