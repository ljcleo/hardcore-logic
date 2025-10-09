from abc import ABC, abstractmethod
from typing import Any

from .common import GenerationConstraint, Reply


class LLMClient(ABC):
    @abstractmethod
    def __init__(self, api_key: str, base_url: str | None, proxy: str | None, timeout: int) -> None:
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def name(cls) -> str:
        raise NotImplementedError()

    @abstractmethod
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
        raise NotImplementedError()

    @abstractmethod
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
        raise NotImplementedError()
