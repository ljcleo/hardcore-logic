from abc import ABC, abstractmethod
from collections.abc import Mapping


class Evaluator(ABC):
    @classmethod
    @abstractmethod
    def task(cls) -> str:
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def check_solution(example: Mapping[str, str]) -> bool:
        raise NotImplementedError()

    def __call__(self, example: Mapping[str, str]) -> int:
        return (
            1 - int(self.check_solution(example))
            if example["error"] is None and example["proposal"] != ""
            else 2
        )
