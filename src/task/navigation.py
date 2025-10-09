from typing import Any, override

from ..common import Example
from .base import GeneralTask
from .common import GeneralResponseModel
from .utils import make_general_regex

type NavigationSolutionType = list[str]
NavigationResponseModel = GeneralResponseModel[NavigationSolutionType]


class Navigation(GeneralTask[NavigationSolutionType]):
    @override
    @classmethod
    def name(cls):
        return "navigation"

    @override
    def prepare_example(
        self, prompt_example: bool, **kwargs: Any
    ) -> Example[NavigationResponseModel]:
        prompt: str = self._make_prompt(prompt_example, puzzle=kwargs["puzzle"])

        regex: str = '"[A-Z]"'
        regex = make_general_regex(rf"\[({regex}(,{regex})*)?\]")

        return {
            "idx": kwargs["id"],
            "prompt": prompt,
            "json_model": NavigationResponseModel,
            "regex": regex,
            "label": kwargs["solution"],
        }
