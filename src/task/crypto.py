from typing import Any, override

from ..common import Example
from .base import GeneralTask
from .common import GeneralResponseModel
from .utils import make_general_regex

type CryptoSolutionType = str
CryptoResponseModel = GeneralResponseModel[CryptoSolutionType]


class Crypto(GeneralTask[CryptoSolutionType]):
    @override
    @classmethod
    def name(cls):
        return "crypto"

    @override
    def prepare_example(self, prompt_example: bool, **kwargs: Any) -> Example[CryptoResponseModel]:
        prompt: str = self._make_prompt(
            prompt_example, puzzle=kwargs["puzzle"], ordered=kwargs["ordered"]
        )

        regex: str = make_general_regex('"[A-Z]+"')

        return {
            "idx": kwargs["id"],
            "prompt": prompt,
            "json_model": CryptoResponseModel,
            "regex": regex,
            "label": kwargs["solution"],
        }
