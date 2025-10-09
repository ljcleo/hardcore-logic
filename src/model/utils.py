from pydantic import BaseModel

from ..llm_client import GenerationConstraint


def make_vllm_constraint(
    *, regex: str | None = None, json_model: type[BaseModel] | None = None
) -> GenerationConstraint | None:
    constraint: GenerationConstraint | None = None

    if regex is not None:
        constraint = {"mode": "regex", "pattern": regex}
    elif json_model is not None:
        constraint = {"mode": "json", "pattern": json_model}

    return constraint
