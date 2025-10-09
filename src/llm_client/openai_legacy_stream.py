import asyncio
import json
import warnings
from typing import Any, override

from httpx import AsyncClient
from openai import AsyncOpenAI
from openai.types import CompletionUsage
from openai.types.chat import (
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ParsedChatCompletion,
)
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionTokensDetails
from pydantic import BaseModel

from .base import GenerationConstraint, LLMClient, Reply


class OpenAILegacyStreamClient(LLMClient):
    @override
    def __init__(self, api_key: str, base_url: str | None, proxy: str | None, timeout: int) -> None:
        self._client: AsyncOpenAI = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=AsyncClient(proxy=proxy, timeout=timeout),
        )

    @override
    @classmethod
    def name(cls) -> str:
        return "openai-legacy-stream"

    @override
    async def chat(
        self,
        prompt: str,
        model: str,
        *,
        max_token: int,
        temperature: float,
        seed: int,
        system_prompt: str | None = None,
        constraint: GenerationConstraint | None = None,
        **kwargs: Any,
    ) -> Reply:
        messages: list[ChatCompletionMessageParam] = [{"role": "user", "content": prompt}]
        if system_prompt is not None:
            messages = [{"role": "system", "content": system_prompt}, *messages]

        if constraint is not None:
            pattern: str | type[BaseModel] = constraint["pattern"]

            if constraint["mode"] == "stop":
                assert isinstance(pattern, str)
                kwargs["stop"] = pattern
            elif constraint["mode"] == "regex":
                assert isinstance(pattern, str)
                kwargs["response_format"] = {"type": "json_object"}

                prompt += (
                    "\n\n**Your output should strictly follow this regular expression:** " + pattern
                )
            elif constraint["mode"] == "json":
                assert not isinstance(pattern, str)
                assert issubclass(pattern, BaseModel)
                kwargs["response_format"] = {"type": "json_object"}

                prompt += (
                    "\n\n**Your output should strictly follow this JSON schema:** "
                    + json.dumps(pattern.model_json_schema(), separators=(",", ":"))
                )
            else:
                raise ValueError(constraint)

        retry: int = 0

        while True:
            try:
                completion: ParsedChatCompletion

                async with self._client.chat.completions.stream(
                    messages=messages,
                    model=model,
                    max_tokens=max_token,
                    temperature=temperature,
                    extra_body=kwargs,
                ) as stream:
                    async for event in stream:
                        if event.type == "chunk":
                            completion = event.snapshot

                completion = stream.current_completion_snapshot
                break
            except Exception as e:
                retry += 1
                if retry > 3:
                    raise

                warnings.warn(f"chat completion failed {retry} times: {repr(e)}")
                await asyncio.sleep(30 * (1 << retry))

        choice: Choice = completion.choices[0]
        completion_message: ChatCompletionMessage = choice.message
        usage: CompletionUsage | None = completion.usage
        output_tokens: int | None = None
        reasoning_tokens: int | None = None

        if usage is not None:
            output_tokens = usage.completion_tokens
            usage_details: CompletionTokensDetails | None = usage.completion_tokens_details

            if usage_details is not None:
                reasoning_tokens = usage_details.reasoning_tokens

        return {
            "stopped": choice.finish_reason == "stop",
            "thought": getattr(completion_message, "reasoning_content", None),
            "response": completion_message.content,
            "output_tokens": output_tokens,
            "reasoning_tokens": reasoning_tokens,
        }

    @override
    async def complete(
        self,
        prompt: str,
        model: str,
        *,
        max_token: int,
        temperature: float,
        seed: int,
        constraint: GenerationConstraint | None = None,
        **kwargs: Any,
    ) -> Reply:
        raise RuntimeError("openai completion is temporarily disabled")
