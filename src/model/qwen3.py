from typing import override

from pydantic import BaseModel

from ..llm_client import LLMClient, Reply
from .base import Model
from .utils import make_vllm_constraint


class Qwen3Model(Model):
    @override
    @classmethod
    def name(cls) -> str:
        return "qwen3"

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
        reply: Reply = await client.chat(
            prompt,
            self.llm_model,
            max_token=self.max_token,
            temperature=self.temperature,
            seed=seed,
            constraint=make_vllm_constraint(regex=regex, json_model=json_model),
        )

        thought: str | None = reply["thought"]
        response: str | None = reply["response"]
        reasoning_tokens: int | None = reply["reasoning_tokens"]
        assert response is not None

        if "<think>" in response:
            thought = response.lstrip("<think>").lstrip()
            response = None
            reasoning_tokens = None

        return {
            "stopped": reply["stopped"],
            "thought": thought,
            "response": response,
            "output_tokens": reply["output_tokens"],
            "reasoning_tokens": reasoning_tokens,
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
            await client.complete(
                f"<|im_start|>user\n{prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n<think>\n{thought}\n</think>\n\n",
                self.llm_model,
                max_token=self.max_comp_token,
                temperature=self.temperature,
                seed=seed,
                constraint=make_vllm_constraint(regex=regex, json_model=json_model),
            )
        )["response"]
