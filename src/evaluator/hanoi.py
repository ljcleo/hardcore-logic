import re
from collections.abc import Mapping
from typing import override

from .base import Evaluator
from .utils import load_json


class HanoiEvaluator(Evaluator):
    @override
    @classmethod
    def task(cls) -> str:
        return "hanoi"

    @override
    @staticmethod
    def check_solution(example: Mapping[str, str]) -> bool:
        idx: str = example["idx"]
        label: str = example["label"]
        proposal: str = example["proposal"]

        if label == "null" or proposal == "null":
            return proposal == label

        step: int = int(label)
        moves: list[tuple[str, str]] = load_json(list[tuple[str, str]], proposal)

        if step > 0 and len(moves) > step:
            return False

        prompt: str = example["prompt"].partition("# Puzzle to Solve\n")[2].strip()

        match: "re.Match[str] | None" = re.search(
            "\\(smallest\\) (`[0-9]+`(?:, `[0-9]+`)*) \\(largest\\)\\.", prompt
        )

        assert match is not None, idx
        order: list[int] = [int(s.strip("` ")) for s in match.group(1).split(",")]
        size_map: dict[int, int] = {k: i for i, k in enumerate(order)}

        match = re.search("## Puzzle to Solve\n([^#]+)\n\n", prompt)
        assert match is not None, idx
        puzzle: list[str] = match.group(1).strip().split("\n")
        n_peg: int = (len(puzzle) >> 1) - 1

        cur: list[list[int]] = [
            [int(c) for c in r.strip().split()[2:]] for r in puzzle[1 : n_peg + 1]
        ]

        goal: list[list[int]] = [
            [int(c) for c in r.strip().split()[2:]] for r in puzzle[n_peg + 2 :]
        ]

        for s, t in moves:
            if len(s) != 1 or len(t) != 1:
                return False

            i: int = ord(s) - ord("A")
            j: int = ord(t) - ord("A")

            if i < 0 or i >= n_peg or j < 0 or j >= n_peg or len(cur[i]) == 0:
                return False

            d: int = cur[i].pop()
            if len(cur[j]) > 0 and size_map[cur[j][-1]] < size_map[d]:
                return False

            cur[j].append(d)

        def to_str(state: list[list[int]]) -> str:
            return "\n".join(" ".join(str(c) for c in r) for r in state)

        return to_str(cur) == to_str(goal)
