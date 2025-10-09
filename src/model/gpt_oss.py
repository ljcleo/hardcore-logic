from typing import override

from pydantic import BaseModel

from ..llm_client import LLMClient, Reply
from .base import Model
from .utils import make_vllm_constraint


class GPTOssModel(Model):
    def __init__(self, model: str, max_token: int, max_comp_token: int, temperature: float) -> None:
        super().__init__(model, max_token, max_comp_token, temperature)

        self.system_prompt: str = (
            "Reasoning: high. Valid channels: analysis, final. "
            "Channel must be included for every message."
        )

    @override
    @classmethod
    def name(cls) -> str:
        return "gpt-oss"

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
            system_prompt=self.system_prompt,
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
                f"<|start|>system<|message|>{self.system_prompt}<|end|>\n"
                f"<|start|>user<|message|>{prompt}<|end|>\n"
                f"<|start|>assistant<|channel|>analysis<|message|>{thought}<|end|>\n"
                "<|start|>assistant<|channel|>final<|message|>",
                self.llm_model,
                max_token=self.max_comp_token,
                temperature=self.temperature,
                seed=seed,
                constraint=make_vllm_constraint(regex=regex, json_model=json_model),
            )
        )["response"]
