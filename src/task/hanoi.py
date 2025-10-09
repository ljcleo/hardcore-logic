from typing import Any, override

from ..common import Example
from ..utils import load_json
from .base import GeneralTask
from .common import GeneralResponseModel
from .utils import make_choice_regex, make_general_regex, make_list_regex

# type HanoiSolutionType = list[tuple[str, str]]
type HanoiSolutionType = list[list[str]]
HanoiResponseModel = GeneralResponseModel[HanoiSolutionType]


class Hanoi(GeneralTask[HanoiSolutionType]):
    @override
    @classmethod
    def name(cls):
        return "hanoi"

    @override
    def prepare_example(self, prompt_example: bool, **kwargs: Any) -> Example[HanoiResponseModel]:
        puzzle: str = kwargs["puzzle"]
        n_peg: int = puzzle.count("|") >> 1
        order: list[int] = load_json(list[int], kwargs["order"])
        n_disk: int = len(order)

        prompt: str = self._make_prompt(
            prompt_example,
            puzzle=puzzle,
            n_peg=n_peg,
            n_disk=n_disk,
            order=order,
            right_only=kwargs["right_only"],
        )

        regex: str = make_choice_regex((chr(ord("A") + i) for i in range(n_peg)), is_str=True)
        regex = make_list_regex((regex, regex))
        regex = make_general_regex(rf"\[({regex}(,{regex}){{0,{1 << n_disk}}})?\]")

        return {
            "idx": kwargs["id"],
            "prompt": prompt,
            "json_model": HanoiResponseModel,
            "regex": regex,
            "label": kwargs["solution"],
        }
