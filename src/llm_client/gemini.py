import asyncio
import warnings
from typing import Any, override

from google.genai import Client
from google.genai.types import (
    Candidate,
    Content,
    FinishReason,
    GenerateContentConfig,
    GenerateContentResponse,
    GenerateContentResponseUsageMetadata,
    HttpOptions,
    Part,
    ThinkingConfig,
)
from pydantic import BaseModel

from .base import GenerationConstraint, LLMClient, Reply


class GeminiClient(LLMClient):
    DYNAMIC_BUDGET: bool = False

    @override
    def __init__(self, api_key: str, base_url: str | None, proxy: str | None, timeout: int) -> None:
        self._client: Client = Client(
            api_key=api_key,
            http_options=HttpOptions(
                base_url=base_url,
                timeout=timeout * 1000,
                client_args={"proxy": proxy},
                async_client_args={"proxy": proxy},
            ),
        )

    @override
    @classmethod
    def name(cls) -> str:
        return "gemini"

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
        config: GenerateContentConfig = GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_token,
            seed=seed,
            thinking_config=ThinkingConfig(
                include_thoughts=True,
                thinking_budget=-1 if self.DYNAMIC_BUDGET else max_token - 128,
            ),
            **kwargs,
        )

        if system_prompt is not None:
            config.system_instruction = system_prompt

        if constraint is not None:
            pattern: str | type[BaseModel] = constraint["pattern"]

            if constraint["mode"] == "stop":
                assert isinstance(pattern, str)
                config.stop_sequences = [pattern]
            elif constraint["mode"] == "regex":
                raise RuntimeError("gemini chat does not support guided regex")
            elif constraint["mode"] == "json":
                assert not isinstance(pattern, str)
                assert issubclass(pattern, BaseModel)
                config.response_mime_type = "application/json"
                config.response_schema = pattern
            else:
                raise ValueError(constraint)

        retry: int = 0

        while True:
            try:
                completion: GenerateContentResponse = (
                    await self._client.aio.models.generate_content(
                        contents=prompt, model=model, config=config
                    )
                )

                break
            except Exception as e:
                retry += 1
                if retry > 3:
                    raise

                warnings.warn(f"chat completion failed {retry} times: {repr(e)}")
                await asyncio.sleep(30 * (1 << retry))

        stopped: bool = False
        thought: str | None = None
        response: str | None = None
        output_tokens: int | None = None
        reasoning_tokens: int | None = None

        candidates: list[Candidate] | None = completion.candidates
        usage_metadata: GenerateContentResponseUsageMetadata | None = completion.usage_metadata

        if candidates is not None:
            candidate: Candidate = candidates[0]
            stopped = candidate.finish_reason == FinishReason.STOP
            content: Content | None = candidate.content

            if content is not None:
                parts: list[Part] | None = content.parts

                if parts is not None:
                    for part in parts:
                        if not part.text:
                            continue
                        if part.thought:
                            thought = part.text
                        else:
                            response = part.text

        if usage_metadata is not None:
            output_tokens = usage_metadata.candidates_token_count
            reasoning_tokens = usage_metadata.thoughts_token_count

            if output_tokens is None:
                output_tokens = reasoning_tokens
            elif reasoning_tokens is not None:
                output_tokens += reasoning_tokens

        return {
            "stopped": stopped,
            "thought": thought,
            "response": response,
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
        raise RuntimeError("gemini does not support completion")


class GeminiDynamicClient(LLMClient):
    DYNAMIC_BUDGET: bool = True

    @override
    @classmethod
    def name(cls) -> str:
        return "gemini-dynamic"
