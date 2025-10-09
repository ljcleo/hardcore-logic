import re
from collections import deque
from collections.abc import Mapping
from typing import override

from .base import Evaluator
from .utils import load_json


class HitoriEvaluator(Evaluator):
    @override
    @classmethod
    def task(cls) -> str:
        return "hitori"

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
        assert match is not None, idx
        puzzle: list[list[str]] = [r.strip().split() for r in match.group(1).split("\n")]
        n: int = len(puzzle)
        locked: list[list[bool]] = [[c.startswith("{") for c in r] for r in puzzle]

        for i in range(n):
            for j in range(n):
                puzzle[i][j] = puzzle[i][j].lstrip("{").rstrip("}")

        if puzzle[0][0].isupper():
            for i in range(n):
                for j in range(n):
                    puzzle[i][j] = chr(ord("1") + (ord(puzzle[i][j]) - ord("A") + n - i) % n)

        def check(x: str) -> bool:
            filled: list[list[bool]] = load_json(list[list[bool]], x)

            for i in range(n):
                for j in range(n):
                    if locked[i][j]:
                        if filled[i][j]:
                            return False

                        adjs: int = 0

                        for di in (-1, 0, 1):
                            for dj in (-1, 0, 1):
                                if di or dj:
                                    ni: int = i + di
                                    nj: int = j + dj

                                    if 0 <= ni < n and 0 <= nj < n:
                                        adjs += filled[ni][nj]

                        if adjs > 3:
                            return False

            for i in range(n):
                if len({puzzle[i][j] for j in range(n) if not filled[i][j]}) != n - sum(filled[i]):
                    return False

                if len({puzzle[j][i] for j in range(n) if not filled[j][i]}) != n - sum(
                    filled[j][i] for j in range(n)
                ):
                    return False

            for i in range(n):
                for j in range(n - 1):
                    if filled[i][j] and filled[i][j + 1]:
                        return False
                    if filled[j][i] and filled[j + 1][i]:
                        return False

            flag: bool = False

            for i in range(n):
                for j in range(n):
                    if not filled[i][j]:
                        if flag:
                            return False
                        else:
                            filled[i][j] = True
                            queue: deque[tuple[int, int]] = deque([(i, j)])
                            delta: list[tuple[int, int]] = [(-1, 0), (1, 0), (0, -1), (0, 1)]

                            while len(queue) > 0:
                                ci: int
                                cj: int
                                ci, cj = queue.popleft()

                                for di, dj in delta:
                                    ni: int = ci + di
                                    nj: int = cj + dj

                                    if 0 <= ni < n and 0 <= nj < n and not filled[ni][nj]:
                                        filled[ni][nj] = True
                                        queue.append((ni, nj))

                            flag = True

            return True

        assert check(label), idx
        return check(proposal)
