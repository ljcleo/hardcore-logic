from argparse import ArgumentParser
from dataclasses import dataclass, field
from typing import get_args

from .llm import GeminiMode, OpenAIMode
from .model import MODELS
from .task import TASKS


@dataclass
class Arguments:
    split: str = ""
    task: str = ""
    api: list[str] = field(default_factory=list)
    model_type: str = ""
    model: str = ""
    prompt_example: bool = False
    run_name: str | None = None
    sample_rep: int = 1
    max_token: int = 32768
    max_comp_token: int = 4096
    temperature: float = 0.0
    seed: int = 19260817
    parse_json: bool = True
    retry_comp: bool = True
    api_rep: int = 1
    api_timeout: int = 600
    openai_mode: OpenAIMode = "default"
    gemini_mode: GeminiMode = "default"
    skip: int | None = None
    count: int | None = None
    select: list[int] | None = None
    continue_run: bool = False

    @staticmethod
    def parse() -> "Arguments":
        parser: ArgumentParser = ArgumentParser()

        parser.add_argument("--split", type=str, required=True)
        parser.add_argument("--task", type=str, choices=TASKS.keys(), required=True, metavar="task")
        parser.add_argument("--api", nargs="+", type=str, required=True)

        parser.add_argument(
            "--model-type", type=str, choices=MODELS.keys(), required=True, metavar="model-type"
        )

        parser.add_argument("--model", type=str, required=True)
        parser.add_argument("--prompt-example", action="store_true")
        parser.add_argument("--run-name", type=str)
        parser.add_argument("--sample-rep", type=int, default=Arguments.sample_rep)
        parser.add_argument("--max-token", type=int, default=Arguments.max_token)
        parser.add_argument("--max-comp-token", type=int, default=Arguments.max_comp_token)
        parser.add_argument("--temperature", type=float, default=Arguments.temperature)
        parser.add_argument("--seed", type=int, default=Arguments.seed)
        parser.add_argument("--no-parse-json", action="store_false", dest="parse_json")
        parser.add_argument("--no-retry-comp", action="store_false", dest="retry_comp")
        parser.add_argument("--api-rep", type=int, default=Arguments.api_rep)
        parser.add_argument("--api-timeout", type=int, default=Arguments.api_timeout)

        parser.add_argument(
            "--openai-mode",
            type=str,
            default=Arguments.openai_mode,
            choices=get_args(OpenAIMode.__value__),
        )

        parser.add_argument(
            "--gemini-mode",
            type=str,
            default=Arguments.gemini_mode,
            choices=get_args(GeminiMode.__value__),
        )

        parser.add_argument("--skip", type=int)
        parser.add_argument("--count", type=int)
        parser.add_argument("--select", nargs="+", type=int)
        parser.add_argument("--continue", action="store_true", dest="continue_run")

        return parser.parse_args(namespace=Arguments())
