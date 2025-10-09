from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from IPython.display import display


class Stat:
    def __init__(self, task: str, run_names: Iterable[str]) -> None:
        self._task: str = task
        self._run_names: list[str] = list(run_names)

    def __call__(self, dataset: str, groups: Iterable[str]) -> pd.DataFrame | None:
        group_list: list[str] = list(groups)
        eval_path: Path = Path("../eval") / self._task
        df: pd.DataFrame | None = None

        for run_name in self._run_names:
            eval_file: Path = eval_path / f"{dataset}_{run_name}.jsonl"

            if eval_file.exists():
                cur_df: pd.DataFrame = self._stat(eval_file, group_list, run_name)

                df = (
                    cur_df
                    if df is None
                    else df.merge(cur_df.iloc[:, 1:], left_index=True, right_index=True)
                )

        return df

    @staticmethod
    def _stat(eval_file: Path, groups: list[str], run_name: str) -> pd.DataFrame:
        group_map: dict[str, int] = {k: i for i, k in enumerate(groups)}
        df: pd.DataFrame = pd.read_json(eval_file, lines=True)
        df["group"] = df["id"].str.rsplit("-", n=1).str[0].map(group_map)

        cnt: pd.Series = df["group"].value_counts(sort=False).sort_index()
        df = df.groupby("group")["verdict"].value_counts(sort=False).unstack().fillna(0).astype(int)

        for i in range(3):
            if i not in df.columns:
                df[i] = 0

        df = cast(pd.DataFrame, df / cast(np.ndarray, cnt.values).reshape(-1, 1)) * 100
        df.sort_index(axis=0, inplace=True)
        df.sort_index(axis=1, inplace=True)
        df = pd.concat([cnt, df], axis=1)
        df.index = pd.Index(groups)

        df.columns = pd.MultiIndex.from_tuples(
            [("", "count"), *((run_name, k) for k in ("right", "wrong", "error"))]
        )

        return df


def display_all_stats(
    dataset: str, model_group: str, task_variants: Mapping[str, Iterable[str]] | None = None
) -> None:
    with open(f"./models/{model_group}.txt", encoding="utf8") as f:
        models: tuple[str, ...] = tuple(r.strip() for r in f)

    groups: pd.Series = (
        pd.read_csv(f"../prepare/list/{dataset}.csv")
        .set_index("task")
        .apply(lambda s: f"{s['dataset']}--{s['group']}", axis=1)
    )

    for task in groups.index.unique():
        if task_variants is not None and task not in task_variants:
            print("skip", task)
            continue

        for variant in (task,) if task_variants is None else task_variants[task]:
            stat: pd.DataFrame | None = Stat(variant, models)(dataset, groups[[task]])

            if stat is not None:
                print(variant)
                display(stat)
