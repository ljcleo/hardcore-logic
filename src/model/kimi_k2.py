from typing import override

from pydantic import BaseModel

from ..llm_client import GenerationConstraint, LLMClient, Reply
from .base import Model
from .utils import make_vllm_constraint


class KimiK2Model(Model):
    def __init__(self, model: str, max_token: int, max_comp_token: int, temperature: float) -> None:
        super().__init__(model, max_token, max_comp_token, temperature)

        self.system_prompt: str = (
            "First output step-by-step reasoning content between <think> and </think>, "
            "then output the final response."
        )

    @override
    @classmethod
    def name(cls) -> str:
        return "kimi-k2"

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
        think_reply: Reply = await self._query_comp_reply(client, prompt, seed=seed, is_comp=False)
        stopped: bool = think_reply["stopped"]
        thought: str | None = think_reply["response"]
        reasoning_tokens: int | None = think_reply["output_tokens"]
        response: str | None = None
        output_tokens: int | None = reasoning_tokens

        if stopped:
            response_reply: Reply = await self._query_comp_reply(
                client,
                prompt,
                seed=seed,
                is_comp=True,
                thought=thought,
                json_model=json_model,
                regex=regex,
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
                client,
                prompt,
                seed=seed,
                is_comp=True,
                thought=thought,
                json_model=json_model,
                regex=regex,
            )
        )["response"]

    async def _query_comp_reply(
        self,
        client: LLMClient,
        prompt: str,
        *,
        seed: int,
        is_comp: bool,
        thought: str | None = None,
        json_model: type[BaseModel] | None = None,
        regex: str | None = None,
    ) -> Reply:
        prompt = (
            f"<|im_system|>system<|im_middle|>{self.system_prompt}<|im_end|>"
            f"<|im_user|>user<|im_middle|>{prompt}<|im_end|>"
            f"<|im_assistant|>assistant<|im_middle|><think>"
        )

        constraint: GenerationConstraint | None = {"mode": "stop", "pattern": "</think>"}
        max_token: int = self.max_token

        if is_comp:
            prompt += f"{thought}</think>\n\n"
            max_token = self.max_comp_token
            constraint = make_vllm_constraint(regex=regex, json_model=json_model)

        return await client.complete(
            prompt,
            self.llm_model,
            max_token=max_token,
            temperature=self.temperature,
            seed=seed,
            constraint=constraint,
        )
