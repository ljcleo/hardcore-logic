from typing import Any, override

from pydantic import BaseModel


class GeneralResponseModel[T](BaseModel):
    solvable: bool
    solution: T | None

    @override
    @classmethod
    def model_parametrized_name(cls, params: tuple[type[Any], ...]) -> str:
        return f"{params[0].__name__}_Task"
