from typing import override

from pydantic import BaseModel

from ..llm_client import GenerationConstraint, LLMClient, Reply
from .base import Model


class GeminiModel(Model):
    @override
    @classmethod
    def name(cls) -> str:
        return "gemini"

    @override
    async def query_full(
        self,
        client: LLMClient,
        prompt: str,
        *,
        seed: int,
        json_model: type[BaseModel] | None,
        regex: str | None,
    ) -> Reply:
        constraint: GenerationConstraint | None = None

        if json_model is not None:
            constraint = {"mode": "json", "pattern": json_model}
        elif regex is not None:
            constraint = {"mode": "regex", "pattern": regex}

        reply: Reply = await client.chat(
            prompt,
            self.llm_model,
            max_token=self.max_token,
            temperature=self.temperature,
            seed=seed,
            constraint=constraint,
        )

        if not reply["stopped"]:
            reply["response"] = None
        if reply["thought"] is None:
            reply["thought"] = f"<<hidden reasoning tokens: {reply['reasoning_tokens']}>>"

        return reply

    @override
    async def query_comp(
        self,
        client: LLMClient,
        prompt: str,
        thought: str | None,
        *,
        seed: int,
        json_model: type[BaseModel] | None,
        regex: str | None,
    ) -> str | None:
        return (
            await self.query_full(client, prompt, seed=seed, json_model=json_model, regex=regex)
        )["response"]
