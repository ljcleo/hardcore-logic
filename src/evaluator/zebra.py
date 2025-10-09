from collections.abc import Mapping
from typing import override

from .base import Evaluator


class ZebraEvaluator(Evaluator):
    @override
    @classmethod
    def task(cls) -> str:
        return "zebra"

    @override
    @staticmethod
    def check_solution(example: Mapping[str, str]) -> bool:
        return example["proposal"] == example["label"]
