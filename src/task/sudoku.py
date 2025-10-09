from typing import Any, override

from ..common import Example
from ..utils import load_json
from .base import GeneralTask
from .common import GeneralResponseModel
from .utils import make_choice_regex, make_general_regex, make_list_regex

type SudokuSolutionType = list[list[str]]
SudokuResponseModel = GeneralResponseModel[SudokuSolutionType]


class Sudoku(GeneralTask[SudokuSolutionType]):
    @override
    @classmethod
    def name(cls):
        return "sudoku"

    @override
    def prepare_example(self, prompt_example: bool, **kwargs: Any) -> Example[SudokuResponseModel]:
        subs: list[str] = load_json(list[str], kwargs["subs"])

        prompt: str = self._make_prompt(
            prompt_example,
            puzzle=kwargs["puzzle"],
            subs=subs,
            diag=kwargs["diag"],
            discon=kwargs["discon"],
            irzone=kwargs["irzone"],
            mc_box=kwargs["mc_box"],
        )

        n: int = len(subs) - 1
        regex: str = make_choice_regex(subs[1:], is_str=True)
        regex = make_list_regex(regex for _ in range(n))
        regex = make_general_regex(make_list_regex(regex for _ in range(n)))

        return {
            "idx": kwargs["id"],
            "prompt": prompt,
            "json_model": SudokuResponseModel,
            "regex": regex,
            "label": kwargs["solution"],
        }
