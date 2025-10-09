import re
from collections.abc import Mapping
from typing import override

from .base import Evaluator
from .utils import load_json


class KakurasuEvaluator(Evaluator):
    @override
    @classmethod
    def task(cls) -> str:
        return "kakurasu"

    @override
    @staticmethod
    def check_solution(example: Mapping[str, str]) -> bool:
        idx: str = example["idx"]
        label: str = example["label"]
        proposal: str = example["proposal"]

        if label == "null" or proposal == "null":
            return proposal == label

        prompt: str = example["prompt"].partition("# Puzzle to Solve\n")[2].strip()
        match: "re.Match[str] | None" = re.search("## Puzzle to Solve\n([^#]+)\n\n", prompt)
        assert match is not None

        raw_puzzle: list[list[str]] = [
            [c.strip() for c in r.strip().split()]
            for r in match.group(1).replace("|", " ").split("\n")
        ]

        block: list[list[bool]] = [[c == "X" for c in r[1:]] for r in raw_puzzle[1:]]
        row_hint: list[int] = [int(r[0]) for r in raw_puzzle[1:]]
        col_hint: list[int] = [int(x) for x in raw_puzzle[0]]

        n_row: int = len(row_hint)
        n_col: int = len(col_hint)

        for r in block:
            assert len(r) == n_col, idx

        def check(x: str) -> bool:
            filled: list[list[bool]] = load_json(list[list[bool]], x)

            for i in range(n_row):
                for j in range(n_col):
                    if block[i][j] and filled[i][j]:
                        return False

            for i in range(n_row):
                if (
                    row_hint[i] != -1
                    and sum(j + 1 for j in range(n_col) if filled[i][j]) != row_hint[i]
                ):
                    return False

            for i in range(n_col):
                if (
                    col_hint[i] != -1
                    and sum(j + 1 for j in range(n_row) if filled[j][i]) != col_hint[i]
                ):
                    return False

            return True

        assert check(label), idx
        return check(proposal)
