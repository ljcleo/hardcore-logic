from typing import override

from pydantic import BaseModel

from ..llm_client import LLMClient, Reply
from .base import Model
from .utils import make_vllm_constraint


class DeepSeekV31Model(Model):
    @override
    @classmethod
    def name(cls) -> str:
        return "deepseek-v31"

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
        think_reply: Reply = await client.complete(
            f"<｜begin▁of▁sentence｜><｜User｜>{prompt}<｜Assistant｜><think>",
            self.llm_model,
            max_token=self.max_token,
            temperature=self.temperature,
            seed=seed,
            constraint={"mode": "stop", "pattern": "</think>"},
        )

        stopped: bool = think_reply["stopped"]
        thought: str | None = think_reply["response"]
        reasoning_tokens: int | None = think_reply["output_tokens"]
        response: str | None = None
        output_tokens: int | None = reasoning_tokens

        if stopped:
            response_reply: Reply = await self._query_comp_reply(
                client, prompt, thought, seed=seed, json_model=json_model, regex=regex
            )

            response = response_reply["response"]
            output_tokens = response_reply["output_tokens"]

            if output_tokens is not None and reasoning_tokens is not None:
                output_tokens += reasoning_tokens

        return {
            "stopped": stopped,
            "thought": thought,
            "response": response,
            "output_tokens": output_tokens,
            "reasoning_tokens": think_reply["output_tokens"],
        }

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
            await self._query_comp_reply(
                client, prompt, thought, seed=seed, json_model=json_model, regex=regex
            )
        )["response"]

    async def _query_comp_reply(
        self,
        client: LLMClient,
        prompt: str,
        thought: str | None,
        *,
        seed: int,
        json_model: type[BaseModel] | None,
        regex: str | None,
    ) -> Reply:
        return await client.complete(
            f"<｜begin▁of▁sentence｜><｜User｜>{prompt}<｜Assistant｜><think>{thought}</think>",
            self.llm_model,
            max_token=self.max_comp_token,
            temperature=self.temperature,
            seed=seed,
            constraint=make_vllm_constraint(regex=regex, json_model=json_model),
        )
