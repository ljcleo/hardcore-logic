from typing import override

from pydantic import BaseModel

from ..llm_client import LLMClient, Reply
from .base import Model
from .utils import make_vllm_constraint


class Qwen3ThinkModel(Model):
    @override
    @classmethod
    def name(cls) -> str:
        return "qwen3-think"

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
        return await client.chat(
            prompt,
            self.llm_model,
            max_token=self.max_token,
            temperature=self.temperature,
            seed=seed,
            constraint=make_vllm_constraint(regex=regex, json_model=json_model),
        )

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
