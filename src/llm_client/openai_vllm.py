import asyncio
import re
import warnings
from typing import Any, override

from httpx import AsyncClient
from openai import AsyncOpenAI, BadRequestError, LengthFinishReasonError
from openai.types import Completion, CompletionChoice, CompletionUsage
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ParsedChatCompletion,
)
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionTokensDetails
from pydantic import BaseModel

from .base import GenerationConstraint, LLMClient, Reply


class OpenAIvLLMClient(LLMClient):
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
        return "openai-vllm"

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

        response_format: type[BaseModel] | None = None

        if constraint is not None:
            pattern: str | type[BaseModel] = constraint["pattern"]

            if constraint["mode"] == "stop":
                assert isinstance(pattern, str)
                kwargs["stop"] = pattern
            elif constraint["mode"] == "regex":
                assert isinstance(pattern, str)
                kwargs["guided_regex"] = pattern
            elif constraint["mode"] == "json":
                assert not isinstance(pattern, str)
                assert issubclass(pattern, BaseModel)
                response_format = pattern
            else:
                raise ValueError(constraint)

        retry: int = 0

        while True:
            try:
                completion: ParsedChatCompletion | ChatCompletion

                if response_format is not None:
                    try:
                        completion = await self._client.chat.completions.parse(
                            messages=messages,
                            model=model,
                            max_completion_tokens=max_token,
                            response_format=response_format,
                            seed=seed,
                            temperature=temperature,
                            extra_body=kwargs,
                        )
                    except LengthFinishReasonError as e:
                        completion = e.completion
                else:
                    completion = await self._client.chat.completions.create(
                        messages=messages,
                        model=model,
                        max_completion_tokens=max_token,
                        seed=seed,
                        temperature=temperature,
                        extra_body=kwargs,
                    )

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
        if constraint is not None:
            pattern: str | type[BaseModel] = constraint["pattern"]

            if constraint["mode"] == "stop":
                assert isinstance(pattern, str)
                kwargs["stop"] = pattern
            elif constraint["mode"] == "regex":
                assert isinstance(pattern, str)
                kwargs["guided_regex"] = pattern
            elif constraint["mode"] == "json":
                assert not isinstance(pattern, str)
                assert issubclass(pattern, BaseModel)
                kwargs["guided_json"] = pattern.model_json_schema()
            else:
                raise ValueError(constraint)

        async def make_completion(n: int) -> Completion:
            retry: int = 0

            while True:
                try:
                    return await self._client.completions.create(
                        prompt=prompt,
                        model=model,
                        max_tokens=n,
                        seed=seed,
                        temperature=temperature,
                        extra_body=kwargs,
                    )
                except Exception as e:
                    retry += 1
                    if retry > 3:
                        raise

                    warnings.warn(f"completion failed {retry} times: {repr(e)}")
                    await asyncio.sleep(30 * (1 << retry))

        completion: Completion

        try:
            completion = await make_completion(max_token)
        except BadRequestError as e:
            if isinstance(e.body, dict):
                m: re.Match | None = re.search(
                    "This model's maximum context length is ([0-9]+) tokens\\. "
                    "However, you requested [0-9]+ tokens "
                    "\\(([0-9]+) in the messages, [0-9]+ in the completion\\)\\.",
                    e.body.get("message", ""),
                )

                if m is not None:
                    n: int = int(m.group(1)) - int(m.group(2))

                    if n <= 0:
                        return {
                            "stopped": False,
                            "thought": None,
                            "response": "",
                            "output_tokens": None,
                            "reasoning_tokens": None,
                        }

                    completion = await make_completion(n)
                else:
                    raise
            else:
                raise

        choice: CompletionChoice = completion.choices[0]
        usage: CompletionUsage | None = completion.usage

        return {
            "stopped": choice.finish_reason == "stop",
            "thought": None,
            "response": completion.choices[0].text,
            "output_tokens": None if usage is None else usage.completion_tokens,
            "reasoning_tokens": None,
        }
