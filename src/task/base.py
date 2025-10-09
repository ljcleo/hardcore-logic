from abc import ABC, abstractmethod
from typing import Any, override

from jinja2 import Template
from pydantic import BaseModel

from ..common import Example, Response
from ..utils import dump_json
from .common import GeneralResponseModel


class Task[T: BaseModel](ABC):
    def __init__(self, template: Template) -> None:
        self.template: Template = template

    @classmethod
    @abstractmethod
    def name(cls) -> str:
        raise NotImplementedError()

    @abstractmethod
    def prepare_example(self, prompt_example: bool, **kwargs: Any) -> Example[T]:
        raise NotImplementedError()

    @abstractmethod
    def parse_response(self, response: str, json_model: type[T]) -> Response:
        raise NotImplementedError()

    def _make_prompt(self, prompt_example: bool, **kwargs: Any) -> str:
        return self.template.render(prompt_example=prompt_example, **kwargs)


class GeneralTask[T](Task[GeneralResponseModel[T]]):
    @override
    def parse_response(self, response: str, json_model: type[GeneralResponseModel[T]]) -> Response:
        parsed_response: GeneralResponseModel[T] = json_model.model_validate_json(response)
        proposal: str = "null"

        if parsed_response.solvable:
            parsed_solution: T | None = parsed_response.solution
            assert parsed_solution is not None
            proposal = dump_json(parsed_solution)

        return {"reasoning": "", "proposal": proposal}


class FlexibleGeneralTask(Task[GeneralResponseModel]):
    @override
    def parse_response[T](
        self, response: str, json_model: type[GeneralResponseModel[T]]
    ) -> Response:
        parsed_response: GeneralResponseModel[T] = json_model.model_validate_json(response)
        proposal: str = "null"

        if parsed_response.solvable:
            parsed_solution: T | None = parsed_response.solution
            assert parsed_solution is not None
            proposal = dump_json(parsed_solution)

        return {"reasoning": "", "proposal": proposal}
