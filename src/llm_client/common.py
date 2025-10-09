from typing import Literal, TypedDict

from pydantic import BaseModel


class GenerationConstraint(TypedDict):
    mode: Literal["stop", "regex", "json"]
    pattern: str | type[BaseModel]


class Reply(TypedDict):
    stopped: bool
    thought: str | None
    response: str | None
    output_tokens: int | None
    reasoning_tokens: int | None
