import re
from collections.abc import Mapping
from typing import override

from .base import Evaluator
from .utils import load_json


class BinarioEvaluator(Evaluator):
    @override
    @classmethod
    def task(cls) -> str:
        return "binario"

    @override
    @staticmethod
    def check_solution(example: Mapping[str, str]) -> bool:
        idx: str = example["idx"]
        label: str = example["label"]
        proposal: str = example["proposal"]

        if label == "null" or proposal == "null":
            return proposal == label

        prompt: str = example["prompt"].partition("# Puzzle to Solve\n")[2].strip()
        match = re.search("## Puzzle to Solve\n([^#]+)\n\n", prompt)
        assert match is not None

        puzzle: list[list[int]] = [
            [-1 if c == "." else int(c) for c in r.strip().split()]
            for r in match.group(1).split("\n")
        ]

        n: int = len(puzzle)
        for row in puzzle:
            assert len(row) == n

        constraint: list[str] = []

        if "### Extra Clues:" in prompt:
            match = re.search("### Extra Clues:\n([^#R]+)\n\n", prompt)
            assert match is not None
            constraint.extend(s.lstrip("- ") for s in match.group(1).split("\n"))

        def check(x: str) -> bool:
            filled: list[list[int]] = load_json(list[list[int]], x)

            for i in range(n):
                for j in range(n):
                    if puzzle[i][j] != -1 and filled[i][j] != puzzle[i][j]:
                        return False

            for i in range(n):
                if sum(filled[i]) != n // 2:
                    return False
                if sum(filled[j][i] for j in range(n)) != n // 2:
                    return False

                for j in range(n - 2):
                    if filled[i][j] == filled[i][j + 1] == filled[i][j + 2]:
                        return False
                    if filled[j][i] == filled[j + 1][i] == filled[j + 2][i]:
                        return False

            for c in constraint:
                a: str
                b: str
                a, _, b = c.partition(" != " if (ne := "!" in c) else " == ")

                def gv(p: str) -> int:
                    return int(
                        p if p.isdigit() else filled[ord(p[0]) - ord("A")][ord(p[1]) - ord("a")]
                    )

                if (sum(gv(p) for p in a.split(" + ")) == sum(gv(p) for p in b.split(" + "))) == ne:
                    return False

            return True

        assert check(label), idx
        return check(proposal)
