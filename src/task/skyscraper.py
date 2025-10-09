from typing import Any, override

from ..common import Example
from .base import GeneralTask
from .common import GeneralResponseModel
from .utils import make_choice_regex, make_general_regex, make_list_regex

type SkyscraperSolutionType = list[list[int]]
SkyscraperResponseModel = GeneralResponseModel[SkyscraperSolutionType]


class Skyscraper(GeneralTask[SkyscraperSolutionType]):
    @override
    @classmethod
    def name(cls):
        return "skyscraper"

    @override
    def prepare_example(
        self, prompt_example: bool, **kwargs: Any
    ) -> Example[SkyscraperResponseModel]:
        puzzle: str = kwargs["puzzle"]
        n: int = puzzle.count("\n") - 1
        prompt: str = self._make_prompt(prompt_example, puzzle=puzzle, n=n, mode=kwargs["mode"])

        regex: str = make_choice_regex(map(str, range(1, n + 1)), is_str=False)
        regex = make_list_regex(regex for _ in range(n))
        regex = make_general_regex(make_list_regex(regex for _ in range(n)))

        return {
            "idx": kwargs["id"],
            "prompt": prompt,
            "json_model": SkyscraperResponseModel,
            "regex": regex,
            "label": kwargs["solution"],
        }
