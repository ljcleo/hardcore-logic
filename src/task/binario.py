from typing import Any, override

from ..common import Example
from .base import GeneralTask
from .common import GeneralResponseModel
from .utils import make_general_regex, make_list_regex

type BinarioSolutionType = list[list[int]]
BinarioResponseModel = GeneralResponseModel[BinarioSolutionType]


class Binario(GeneralTask[BinarioSolutionType]):
    @override
    @classmethod
    def name(cls):
        return "binario"

    @override
    def prepare_example(self, prompt_example: bool, **kwargs: Any) -> Example[BinarioResponseModel]:
        puzzle: str = kwargs["puzzle"]
        n: int = puzzle.partition("\n")[0].count(" ") + 1
        prompt: str = self._make_prompt(prompt_example, puzzle=puzzle, n=n)

        regex: str = make_list_regex("(0|1)" for _ in range(n))
        regex = make_general_regex(make_list_regex(regex for _ in range(n)))

        return {
            "idx": kwargs["id"],
            "prompt": prompt,
            "json_model": BinarioResponseModel,
            "regex": regex,
            "label": kwargs["solution"],
        }
