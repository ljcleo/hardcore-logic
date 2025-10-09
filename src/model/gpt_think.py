import warnings
from typing import Literal, override

from pydantic import BaseModel

from ..llm_client import GenerationConstraint, LLMClient, Reply
from .base import Model


class GPTThinkModel(Model):
    REASONING_EFFORT: Literal["minimum", "low", "medium", "high"] | None = None

    def __init__(self, model: str, max_token: int, max_comp_token: int, temperature: float) -> None:
        if temperature != 1.0:
            warnings.warn("[gpt-think] This model only supports temperature = 1.0!")
            temperature = 1.0

        super().__init__(model, max_token, max_comp_token, temperature)

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
            reasoning_effort=self.REASONING_EFFORT,
        )

        if not reply["stopped"]:
            reply["response"] = None

        if reply["thought"] is None:
            reply["thought"] = f"<<hidden reasoning tokens: {reply['reasoning_tokens']}>>"
        elif reply["response"] is not None and reply["response"].startswith(reply["thought"]):
            reply["response"] = reply["response"][len(reply["thought"]) :]

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


class GPTThinkLowModel(GPTThinkModel):
    REASONING_EFFORT = "low"

    @override
    @classmethod
    def name(cls) -> str:
        return "gpt-think-low"


class GPTThinkMediumModel(GPTThinkModel):
    REASONING_EFFORT = "medium"

    @override
    @classmethod
    def name(cls) -> str:
        return "gpt-think-medium"


class GPTThinkHighModel(GPTThinkModel):
    REASONING_EFFORT = "high"

    @override
    @classmethod
    def name(cls) -> str:
        return "gpt-think-high"
