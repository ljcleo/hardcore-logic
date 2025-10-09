from abc import ABC, abstractmethod

from pydantic import BaseModel

from ..llm_client import LLMClient, Reply


class Model(ABC):
    def __init__(self, model: str, max_token: int, max_comp_token: int, temperature: float) -> None:
        self.llm_model: str = model
        self.max_token: int = max_token
        self.max_comp_token: int = max_comp_token
        self.temperature: float = temperature

    @classmethod
    @abstractmethod
    def name(cls) -> str:
        raise NotImplementedError()

    @abstractmethod
    async def query_full(
        self,
        client: LLMClient,
        prompt: str,
        *,
        seed: int,
        json_model: type[BaseModel] | None,
        regex: str | None,
    ) -> Reply:
        raise NotImplementedError()

    @abstractmethod
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
        raise NotImplementedError()
