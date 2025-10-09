from argparse import ArgumentParser
from collections import defaultdict
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class CommandArgs:
    name: str = ""
    ref: list[str] = field(default_factory=list)
    ref_do_sample: bool = False

    @staticmethod
    def parse() -> "CommandArgs":
        parser: ArgumentParser = ArgumentParser()
        parser.add_argument("name", type=str)
        parser.add_argument("--ref", nargs="+", type=str)
        parser.add_argument("--ref-do-sample", action="store_true")
        return parser.parse_args(namespace=CommandArgs())


def main():
    prepare_path: Path = Path("./prepare")
    data_path: Path = prepare_path / "data"
    list_path: Path = prepare_path / "list"
    output_path: Path = Path("./data")

    args: CommandArgs = CommandArgs.parse()
    print(args)

    group_list: pd.DataFrame = pd.read_csv(list_path / f"{args.name}.csv")
    task_dfs: defaultdict[str, list[pd.DataFrame]] = defaultdict(list)
    task: str
    dataset: str
    group: str
    n_sample: int

    for task, dataset, group, n_sample in group_list.itertuples(index=False):
        key: str = f"{task}_{dataset}_{group}"
        seed: int = int(sha256(key.encode("utf8")).hexdigest(), base=16) & ((1 << 32) - 1)
        print(key, "seed", seed, "ref", args.ref)
        df: pd.DataFrame

        if len(args.ref) > 0:
            df = pd.concat(
                [
                    pd.read_parquet(p).query(f"id.str.startswith('{dataset}--{group}-')")
                    for p in filter(
                        Path.exists, (output_path / task / f"{ref}.parquet" for ref in args.ref)
                    )
                ]
            )

            if n_sample >= 0:
                if args.ref_do_sample:
                    df = df.sample(n_sample, random_state=np.random.default_rng(seed=seed))
                else:
                    assert df.shape[0] == n_sample
        else:
            df = pd.read_parquet(data_path / task / f"{dataset}.parquet").query(
                f"id.str.startswith('{group}-')"
            )

            if n_sample >= 0:
                df = df.sample(n_sample, random_state=np.random.default_rng(seed=seed))

            df["id"] = df["id"].map(lambda x: f"{dataset}--{x}")

        task_dfs[task].append(df)

    for task, dfs in task_dfs.items():
        df = pd.concat(dfs, ignore_index=True)
        print(task, df.shape[0])

        output_file: Path = output_path / task / f"{args.name}.parquet"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_file)


if __name__ == "__main__":
    main()
