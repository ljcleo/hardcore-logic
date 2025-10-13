import asyncio
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, NoReturn, cast

from datasets import Dataset, load_dataset
from jinja2 import Environment, FileSystemLoader, select_autoescape
from pydantic import BaseModel
from tqdm import tqdm

from .args import Arguments
from .common import CaseResult, Example, Response
from .llm import LLM, GeminiMode, OpenAIMode
from .llm_client import Reply
from .model import MODELS, Model
from .task import TASKS, Task


class Main:
    def __init__(self) -> None:
        args: Arguments = Arguments().parse()
        print(args)
        self._init_processor(task=args.task)

        self._init_data(
            split=args.split,
            task=args.task,
            prompt_example=args.prompt_example,
            skip=args.skip,
            count=args.count,
            select=args.select,
        )

        self._init_llm(
            api_list=args.api,
            api_rep=args.api_rep,
            api_timeout=args.api_timeout,
            openai_mode=args.openai_mode,
            gemini_mode=args.gemini_mode,
            model_type=args.model_type,
            model=args.model,
            max_token=args.max_token,
            max_comp_token=args.max_comp_token,
            temperature=args.temperature,
            seed=args.seed,
            parse_json=args.parse_json,
            retry_comp=args.retry_comp,
        )

        self._init_output(
            task=args.task,
            sub_task=args.split,
            run_name=args.run_name,
            continue_task=args.continue_run,
        )

        self._init_pbar(sample_rep=args.sample_rep)

    async def execute(self) -> None:
        await asyncio.gather(
            *(
                self._process(example, run_id)
                for example in self.examples
                for run_id in range(self.sample_rep)
            )
        )

    def _init_processor(self, *, task: str) -> None:
        self.processor: Task = TASKS[task](
            Environment(
                loader=FileSystemLoader("./template"), autoescape=select_autoescape()
            ).get_template(f"{task}.md.jinja2")
        )

    def _init_data(
        self,
        *,
        split: str,
        task: str,
        prompt_example: bool,
        skip: int | None,
        count: int | None,
        select: list[int] | None,
    ) -> None:
        with TemporaryDirectory() as cache_dir:
            dataset: Dataset = cast(
                Dataset,
                load_dataset(
                    "parquet",
                    data_files={"test": f"./data/{split}/{task}_{split}.parquet"},
                    split="test",
                    cache_dir=cache_dir,
                ),
            )

            if select is None:
                if skip is not None:
                    dataset = dataset.skip(skip)
                if count is not None:
                    dataset = dataset.take(count)
            else:
                dataset = dataset.select(select)

            self.examples: list[Example] = [
                self.processor.prepare_example(prompt_example, **cast(dict[str, Any], example))
                for example in dataset
            ]

    def _init_llm(
        self,
        *,
        api_list: list[str],
        api_rep: int,
        api_timeout: int,
        openai_mode: OpenAIMode,
        gemini_mode: GeminiMode,
        model_type: str,
        model: str,
        max_token: int,
        max_comp_token: int,
        temperature: float,
        seed: int,
        parse_json: bool,
        retry_comp: bool,
    ) -> None:
        self.llm: LLM = LLM(
            Path("./config/api.json"), api_list, api_rep, api_timeout, openai_mode, gemini_mode
        )

        self.model: Model = MODELS[model_type](model, max_token, max_comp_token, temperature)
        self.seed: int = seed
        self.parse_json: bool = parse_json
        self.retry_comp: bool = retry_comp

    def _init_output(
        self, *, task: str, sub_task: str, run_name: str | None, continue_task: bool
    ) -> None:
        output_dir: Path = Path(f"./output/{task}")
        output_dir.mkdir(parents=True, exist_ok=True)

        self.output_file: Path = (
            output_dir / f"{sub_task}{'' if run_name is None else f'_{run_name}'}.jsonl"
        )

        if not continue_task:
            self.output_file.write_bytes(b"")

        self.output_lock: asyncio.Lock = asyncio.Lock()

    def _init_pbar(self, *, sample_rep: int) -> None:
        self.sample_rep: int = sample_rep

        self.pbar: "tqdm[NoReturn]" = tqdm(
            None, total=len(self.examples) * sample_rep, dynamic_ncols=True
        )

    async def _process(self, example: Example, run_id: int) -> None:
        idx: str = example["idx"]
        prompt: str = example["prompt"]
        json_model: type[BaseModel] = example["json_model"]
        query_json_model: type[BaseModel] | None = json_model if self.parse_json else None
        regex: str = example["regex"]
        label: str = example["label"]
        seed: int = self.seed + run_id

        result: CaseResult = CaseResult(
            idx=idx, prompt=prompt, regex=regex, label=label, run=run_id
        )

        try:
            async with self.llm.get_client() as client:
                completion: Reply = await self.model.query_full(
                    client,
                    prompt,
                    seed=seed,
                    json_model=query_json_model,
                    regex=regex,
                )

                result.thought = completion["thought"]
                result.output = completion["response"]

                if result.output is None:
                    result.reasoning = ""
                    result.proposal = ""
                else:

                    def parse_output(output: str) -> None:
                        result.output = output

                        response: Response = self.processor.parse_response(
                            result.output, json_model
                        )

                        result.reasoning = response["reasoning"]
                        result.proposal = response["proposal"]

                    try:
                        parse_output(result.output)
                    except Exception:
                        if not self.retry_comp:
                            raise

                        output: str | None = await self.model.query_comp(
                            client,
                            prompt,
                            result.thought,
                            seed=seed,
                            json_model=query_json_model,
                            regex=regex,
                        )

                        assert output is not None
                        parse_output(output)
        except Exception as e:
            result.error = repr(e)
            warnings.warn(f"{idx} ({run_id}): {result.error}")

        async with self.output_lock:
            with self.output_file.open("a", encoding="utf8") as f:
                f.write(f"{result.model_dump_json()}\n")

        self.pbar.update()


def main() -> None:
    asyncio.run(Main().execute())
