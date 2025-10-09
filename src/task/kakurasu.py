from typing import Any, override

from ..common import Example
from .base import GeneralTask
from .common import GeneralResponseModel
from .utils import make_general_regex, make_list_regex

type KakurasuSolutionType = list[list[bool]]
KakurasuResponseModel = GeneralResponseModel[KakurasuSolutionType]


class Kakurasu(GeneralTask[KakurasuSolutionType]):
    @override
    @classmethod
    def name(cls):
        return "kakurasu"

    @override
    def prepare_example(
        self, prompt_example: bool, **kwargs: Any
    ) -> Example[KakurasuResponseModel]:
        puzzle: str = kwargs["puzzle"]
        n_row: int = puzzle.count("\n")
        n_col: int = len(puzzle.partition("\n")[0].strip().split())
        prompt: str = self._make_prompt(prompt_example, puzzle=puzzle, n_row=n_row, n_col=n_col)

        regex = make_list_regex("(true|false)" for _ in range(n_col))
        regex = make_general_regex(make_list_regex(regex for _ in range(n_row)))

        return {
            "idx": kwargs["id"],
            "prompt": prompt,
            "json_model": KakurasuResponseModel,
            "regex": regex,
            "label": kwargs["solution"],
        }
