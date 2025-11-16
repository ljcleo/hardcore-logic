from collections.abc import Mapping
from typing import override

from .base import Evaluator
from .utils import compress_json


class CryptoEvaluator(Evaluator):
    @override
    @classmethod
    def task(cls) -> str:
        return "crypto"

    @override
    @staticmethod
    def check_solution(example: Mapping[str, str]) -> bool:
        return compress_json(example["proposal"]) == compress_json(example["label"])
