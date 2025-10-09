import re
from collections import deque
from collections.abc import Mapping
from typing import override

from .base import Evaluator
from .utils import load_json


class SudokuEvaluator(Evaluator):
    @override
    @classmethod
    def task(cls) -> str:
        return "sudoku"

    @override
    @staticmethod
    def check_solution(example: Mapping[str, str]) -> bool:
        idx: str = example["idx"]
        label: str = example["label"]
        proposal: str = example["proposal"]

        if label == "null" or proposal == "null":
            return proposal == label

        prompt: str = example["prompt"].partition("# Puzzle to Solve\n")[2].strip()
        match: "re.Match[str] | None" = re.search("elements: (`[^`]+`(?:, `[^`]+`)*)\\.", prompt)
        assert match is not None, idx
        subs: list[str] = [s.strip("` ") for s in match.group(1).split(",")]
        n: int = len(subs)
        subs_set: set[str] = set(subs)
        subs_map: dict[str, int] = {k: i + 1 for i, k in enumerate(sorted(subs))}

        diag_flag: bool = "EXTRA: Each candidate" in prompt
        discon_flag: bool = "EXTRA: Adjacent cells" in prompt
        mc_box: int = -1

        if "EXTRA: The score of a zone" in prompt:
            match = re.search("zone ([0-9]+) has the highest score", prompt)
            assert match is not None, idx
            mc_box = int(match.group(1)) - 1

        match = re.search("## Puzzle to Solve\n([^#]+)\n\n", prompt)
        assert match is not None, idx
        raw_puzzle: list[str] = [r[1:] for r in match.group(1).split("\n")[1:]]
        puzzle: list[list[str]] = [re.split("[@ ]+", row[1:-1].strip()) for row in raw_puzzle[1::2]]

        assert len(puzzle) == n, idx
        boxes: list[list[tuple[int, int]]] = []
        box_id: dict[tuple[int, int], int] = {}

        for i in range(n):
            for j in range(n):
                p: tuple[int, int] = i, j

                if p not in box_id:
                    k: int = len(boxes)
                    box_id[p] = k
                    boxes.append([p])
                    queue: deque[tuple[int, int]] = deque([p])

                    def mark(pi: int, pj: int):
                        p: tuple[int, int] = pi, pj

                        if p not in box_id:
                            box_id[p] = k
                            boxes[k].append(p)
                            queue.append(p)

                    while len(queue) > 0:
                        ci: int
                        cj: int
                        ci, cj = queue.popleft()

                        if ci > 0 and raw_puzzle[ci * 2][cj * 3 + 1] == " ":
                            mark(ci - 1, cj)
                        if ci < n - 1 and raw_puzzle[ci * 2 + 2][cj * 3 + 1] == " ":
                            mark(ci + 1, cj)
                        if cj > 0 and raw_puzzle[ci * 2 + 1][cj * 3] == " ":
                            mark(ci, cj - 1)
                        if cj < n - 1 and raw_puzzle[ci * 2 + 1][cj * 3 + 3] == " ":
                            mark(ci, cj + 1)

        assert len(boxes) == n, idx
        for box in boxes:
            assert len(box) == n, idx

        def check(x: str) -> bool:
            filled: list[list[str]] = load_json(list[list[str]], x)

            for i in range(n):
                for j in range(n):
                    if puzzle[i][j] != "." and puzzle[i][j] != filled[i][j]:
                        return False

            for i in range(n):
                if set(filled[i]) != subs_set:
                    return False
                if set(filled[j][i] for j in range(n)) != subs_set:
                    return False

            for box in boxes:
                if set(filled[i][j] for i, j in box) != subs_set:
                    return False

            if diag_flag:
                if set(filled[i][i] for i in range(n)) != subs_set:
                    return False
                if set(filled[i][n - i - 1] for i in range(n)) != subs_set:
                    return False

            if discon_flag:
                for i in range(n - 1):
                    for j in range(n):
                        if abs(subs_map[filled[i][j]] - subs_map[filled[i + 1][j]]) <= 1:
                            return False

                for i in range(n):
                    for j in range(n - 1):
                        if abs(subs_map[filled[i][j]] - subs_map[filled[i][j + 1]]) <= 1:
                            return False

            if mc_box >= 0:
                k: int = round(n**0.5)
                kr: int = mc_box // k * k
                kc: int = mc_box % k * k
                assert box_id[kr, kc] == mc_box
                scores: list[int] = [0 for _ in range(n)]

                for i in range(n):
                    for j in range(n):
                        scores[box_id[i, j]] += subs_map[filled[i][j]] * (i % k * k + j % k)

                if scores[mc_box] < max(scores):
                    return False

            return True

        assert check(label), idx
        return check(proposal)
