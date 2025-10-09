from typing import TypedDict

from pydantic import BaseModel


class Example[T: BaseModel](TypedDict):
    idx: str
    prompt: str
    json_model: type[T]
    regex: str
    label: str


class Response(TypedDict):
    reasoning: str
    proposal: str


class CaseResult(BaseModel):
    idx: str
    prompt: str
    regex: str
    label: str
    run: int
    thought: str | None = None
    output: str | None = None
    reasoning: str | None = None
    proposal: str | None = None
    error: str | None = None
