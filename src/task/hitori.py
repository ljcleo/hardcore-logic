from typing import Any, override

from ..common import Example
from .base import GeneralTask
from .common import GeneralResponseModel
from .utils import make_general_regex, make_list_regex

type HitoriSolutionType = list[list[bool]]
HitoriResponseModel = GeneralResponseModel[HitoriSolutionType]


class Hitori(GeneralTask[HitoriSolutionType]):
    @override
    @classmethod
    def name(cls):
        return "hitori"

    @override
    def prepare_example(self, prompt_example: bool, **kwargs: Any) -> Example[HitoriResponseModel]:
        puzzle: str = kwargs["puzzle"]
        n: int = puzzle.count("\n") + 1

        prompt: str = self._make_prompt(
            prompt_example, puzzle=puzzle, n=n, encrypted=kwargs["encrypted"]
        )

        regex = make_list_regex("(true|false)" for _ in range(n))
        regex = make_general_regex(make_list_regex(regex for _ in range(n)))

        return {
            "idx": kwargs["id"],
            "prompt": prompt,
            "json_model": HitoriResponseModel,
            "regex": regex,
            "label": kwargs["solution"],
        }
