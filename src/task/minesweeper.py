from typing import Any, override

from ..common import Example
from .base import GeneralTask
from .common import GeneralResponseModel
from .utils import make_general_regex, make_list_regex

type MinesweeperSolutionType = list[list[bool]]
MinesweeperResponseModel = GeneralResponseModel[MinesweeperSolutionType]


class Minesweeper(GeneralTask[MinesweeperSolutionType]):
    @override
    @classmethod
    def name(cls):
        return "minesweeper"

    @override
    def prepare_example(
        self, prompt_example: bool, **kwargs: Any
    ) -> Example[MinesweeperResponseModel]:
        puzzle: str = kwargs["puzzle"]
        n_row = puzzle.count("\n") + 1
        n_col = puzzle.count(" ") // n_row + 1

        prompt: str = self._make_prompt(
            prompt_example,
            puzzle=puzzle,
            n_row=n_row,
            n_col=n_col,
            no_adj=kwargs["no_adj"],
            letter=kwargs["letter"],
            regional=kwargs["regional"],
        )

        regex = make_list_regex("(true|false)" for _ in range(n_col))
        regex = make_general_regex(make_list_regex(regex for _ in range(n_row)))

        return {
            "idx": kwargs["id"],
            "prompt": prompt,
            "json_model": MinesweeperResponseModel,
            "regex": regex,
            "label": kwargs["solution"],
        }
