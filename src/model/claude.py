import re
from typing import Literal, override

from pydantic import BaseModel

from ..llm_client import GenerationConstraint, LLMClient, Reply
from .base import Model


class ClaudeModel(Model):
    REASONING_EFFORT: Literal["minimum", "low", "medium", "high"] | None = None

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

        if reply["response"] is not None and reply["response"].startswith("<think>"):
            reply["response"] = reply["response"].partition("<think>")[2]
            reply["thought"], _, reply["response"] = reply["response"].partition("</think>")

            if reply["response"] == "":
                reply["response"] = None
            else:
                reply["response"] = reply["response"].rsplit("\n\n", maxsplit=1)[-1]
                m: re.Match | None = re.search(r"\{[\s\S]*\}", reply["response"])

                if m is not None:
                    reply["response"] = m.group(0)

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


class ClaudeLowModel(ClaudeModel):
    REASONING_EFFORT = "low"

    @override
    @classmethod
    def name(cls) -> str:
        return "claude-low"


class ClaudeMediumModel(ClaudeModel):
    REASONING_EFFORT = "medium"

    @override
    @classmethod
    def name(cls) -> str:
        return "claude-medium"


class ClaudeHighModel(ClaudeModel):
    REASONING_EFFORT = "high"

    @override
    @classmethod
    def name(cls) -> str:
        return "claude-high"
