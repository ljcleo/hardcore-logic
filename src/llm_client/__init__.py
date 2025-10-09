from .base import LLMClient
from .common import GenerationConstraint, Reply
from .gemini import GeminiClient, GeminiDynamicClient
from .openai_legacy import OpenAILegacyClient
from .openai_legacy_stream import OpenAILegacyStreamClient
from .openai_std import OpenAIClient
from .openai_stream import OpenAIStreamClient
from .openai_vllm import OpenAIvLLMClient

LLM_CLIENTS: dict[str, type[LLMClient]] = {
    client_cls.name(): client_cls
    for client_cls in (
        OpenAIClient,
        OpenAIvLLMClient,
        OpenAIStreamClient,
        OpenAILegacyClient,
        OpenAILegacyStreamClient,
        GeminiClient,
        GeminiDynamicClient,
    )
}

__all__ = ["GenerationConstraint", "Reply", "LLMClient", "LLM_CLIENTS"]
