import re
from collections.abc import Iterable, Mapping
from typing import override

from .base import Evaluator
from .utils import load_json


class SkyscraperEvaluator(Evaluator):
    @override
    @classmethod
    def task(cls) -> str:
        return "skyscraper"

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
        raw_puzzle: list[str] = [r.strip() for r in match.group(1).replace("|", " ").split("\n")]
        up_hint: list[int] = [int(x) for x in raw_puzzle[0].split()]
        down_hint: list[int] = [int(x) for x in raw_puzzle[-1].split()]
        left_hint: list[int] = [int(r.split(maxsplit=1)[0]) for r in raw_puzzle[1:-1]]
        right_hint: list[int] = [int(r.rsplit(maxsplit=1)[-1]) for r in raw_puzzle[1:-1]]

        n: int = len(left_hint)
        assert len(right_hint) == n, idx
        subs: set[int] = set(range(1, n + 1))
        diag_hint: tuple[tuple[int, int], tuple[int, int]] | None = None

        if len(up_hint) > n:
            diag_hint = (up_hint[0], down_hint[-1]), (down_hint[0], up_hint[-1])
            up_hint = up_hint[1:-1]
            down_hint = down_hint[1:-1]

        assert len(up_hint) == n, idx
        assert len(down_hint) == n, idx

        if "count" in prompt:

            def f(i: int, v: int) -> int:
                return 1
        elif "height sum of" in prompt:

            def f(i: int, v: int) -> int:
                return v
        elif "height*" in prompt:

            def f(i: int, v: int) -> int:
                return i * v
        else:
            raise AssertionError(idx)

        def calc(vs: Iterable[int]) -> int:
            r: int = 0
            m: int = -1
            c: int = 0

            for v in vs:
                if v > m:
                    c += 1
                    r += f(c, v)
                    m = v

            return r

        def check(x: str) -> bool:
            filled: list[list[int]] = load_json(list[list[int]], x)

            for i in range(n):
                if set(filled[i]) != subs:
                    return False
                if left_hint[i] != -1 and calc(filled[i]) != left_hint[i]:
                    return False
                if right_hint[i] != -1 and calc(filled[i][::-1]) != right_hint[i]:
                    return False
                if set(filled[j][i] for j in range(n)) != subs:
                    return False
                if up_hint[i] != -1 and calc(filled[j][i] for j in range(n)) != up_hint[i]:
                    return False
                if (
                    down_hint[i] != -1
                    and calc(filled[j][i] for j in range(n - 1, -1, -1)) != down_hint[i]
                ):
                    return False

            if diag_hint is not None:
                if (
                    diag_hint[0][0] != -1
                    and calc(filled[i][i] for i in range(n)) != diag_hint[0][0]
                ):
                    return False
                if (
                    diag_hint[0][1] != -1
                    and calc(filled[i][i] for i in range(n - 1, -1, -1)) != diag_hint[0][1]
                ):
                    return False
                if (
                    diag_hint[1][0] != -1
                    and calc(filled[n - i - 1][i] for i in range(n)) != diag_hint[1][0]
                ):
                    return False
                if (
                    diag_hint[1][1] != -1
                    and calc(filled[n - i - 1][i] for i in range(n - 1, -1, -1)) != diag_hint[1][1]
                ):
                    return False

            return True

        assert check(label), idx
        return check(proposal)
