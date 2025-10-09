import json
from argparse import ArgumentParser
from dataclasses import dataclass, field
from pathlib import Path

from src.evaluator import EVALUATORS, Evaluator


@dataclass
class Arguments:
    task: str = ""
    files: list[str] = field(default_factory=list)
    rewrite: bool = False

    @staticmethod
    def parse() -> "Arguments":
        parser: ArgumentParser = ArgumentParser()
        parser.add_argument("task", type=str, choices=EVALUATORS.keys(), metavar="task")
        parser.add_argument("files", nargs="*", type=str)
        parser.add_argument("--rewrite", action="store_true")
        return parser.parse_args(namespace=Arguments())


def process(in_file: Path, evaluator: Evaluator, out_file: Path, rewrite: bool) -> None:
    if not rewrite and out_file.exists():
        print(f"skip {out_file.stem}")
        return

    print(f"eval {out_file.stem}")
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with in_file.open(encoding="utf8") as f:
        with out_file.open("w", encoding="utf8") as g:
            for row in f:
                example: dict[str, str] = json.loads(row)

                print(
                    json.dumps(
                        {
                            "id": example["idx"],
                            "run": example["run"],
                            "verdict": evaluator(example),
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    sep="\n",
                    file=g,
                )


def main() -> None:
    args: Arguments = Arguments.parse()
    print(args)

    output_path: Path = Path("./output") / args.task
    if not output_path.exists():
        return

    if len(args.files) == 0:
        args.files = sorted(p.stem for p in output_path.iterdir())

    evaluator: Evaluator = EVALUATORS[args.task]()
    eval_path: Path = Path("./eval") / args.task

    for file in args.files:
        process(output_path / f"{file}.jsonl", evaluator, eval_path / f"{file}.jsonl", args.rewrite)


if __name__ == "__main__":
    main()
