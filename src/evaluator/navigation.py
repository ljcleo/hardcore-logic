import re
from collections.abc import Mapping
from typing import override

from .base import Evaluator
from .utils import load_json


class NavigationEvaluator(Evaluator):
    @override
    @classmethod
    def task(cls) -> str:
        return "navigation"

    @override
    @staticmethod
    def check_solution(example: Mapping[str, str]) -> bool:
        idx: str = example["idx"]
        label: str = example["label"]
        proposal: str = example["proposal"]

        if label == "null" or proposal == "null":
            return proposal == label

        len_route: int = int(label)
        proposed_points: list[str] = load_json(list[str], proposal)
        prompt: str = example["prompt"].partition("# Puzzle to Solve\n")[2].strip()
        v_map: dict[str, str] = {}
        e_map: dict[tuple[str, str], int] = {}

        for m in re.finditer(
            "There is a road which is ([0-9]+) meters long "
            "from ([a-z]+) ([A-Z]) to ([a-z]+) ([A-Z])",
            prompt,
        ):
            len_road: int = int(m.group(1))
            src_type: str = m.group(2)
            src_name: str = m.group(3)
            tgt_type: str = m.group(4)
            tgt_name: str = m.group(5)

            if src_name not in v_map:
                v_map[src_name] = src_type
            else:
                assert v_map[src_name] == src_type

            if tgt_name not in v_map:
                v_map[tgt_name] = tgt_type
            else:
                assert v_map[tgt_name] == tgt_type

            rp: tuple[str, str] = src_name, tgt_name
            e_map[rp] = min(len_road, e_map.get(rp, 1 << 31))

        match: "re.Match[str] | None" = re.search("The start point is ([a-z]+) ([A-Z])", prompt)
        assert match is not None, idx
        st_type: str = match.group(1)
        st_name: str = match.group(2)
        assert v_map.get(st_name, "") == st_type, idx

        if proposed_points[0] != st_name:
            return False

        match = re.search(
            "From the start point, how to reach ((?:a|the nearest) [a-z]+"
            "(?: other than the start point)?(?:, and then a [a-z]+)*) in the shortest way\\?",
            prompt,
        )

        assert match is not None, idx

        q_types: list[str] = [
            q.strip().rsplit(" ", maxsplit=1)[-1]
            for q in match.group(1).replace(" other than the start point", "").strip().split(",")
        ]

        len_proposal: int = 0
        q_pivot: int = 0

        for i in range(1, len(proposed_points)):
            if q_pivot >= len(q_types):
                return False

            rp: tuple[str, str] = proposed_points[i - 1], proposed_points[i]
            if rp not in e_map:
                return False

            len_proposal += e_map[rp]
            q_pivot += v_map[proposed_points[i]] == q_types[q_pivot]

        return q_pivot == len(q_types) and len_proposal <= len_route
