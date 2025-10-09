from typing import Any, override

from pydantic import BaseModel

from ..common import Example
from ..utils import load_json, make_model
from .base import FlexibleGeneralTask
from .common import GeneralResponseModel
from .utils import make_choice_regex, make_dict_regex, make_general_regex


class Zebra(FlexibleGeneralTask):
    @override
    @classmethod
    def name(cls):
        return "zebra"

    @override
    def prepare_example(self, prompt_example: bool, **kwargs: Any) -> Example[GeneralResponseModel]:
        house_ids: list[int] = load_json(list[int], kwargs["house_ids"])
        keys: list[str] = load_json(list[str], kwargs["keys"])
        house_alias: str = kwargs["house_alias"]

        house_info_model: type[BaseModel] = make_model(
            f"{house_alias}Info", **{key: str for key in keys}
        )

        per_house_model: type[BaseModel] = make_model(
            f"Per{house_alias}", **{f"{house_alias} {i}": house_info_model for i in house_ids}
        )

        solution: BaseModel = per_house_model.model_validate_json(kwargs["solution"])
        candidates: dict[str, list[str]] = {key: [] for key in keys}

        for i in house_ids:
            house_info: BaseModel = getattr(solution, f"{house_alias} {i}")
            for key, ls in candidates.items():
                ls.append(getattr(house_info, key))

        prompt: str = self._make_prompt(
            prompt_example,
            puzzle=kwargs["puzzle"],
            house_ids=house_ids,
            keys=keys,
            house_alias=house_alias,
        )

        regex: str = make_dict_regex(
            {k: make_choice_regex(vs, is_str=True) for k, vs in candidates.items()}
        )

        regex = make_general_regex(
            make_dict_regex({f"{house_alias} {i}": regex for i in house_ids})
        )

        return {
            "idx": kwargs["id"],
            "prompt": prompt,
            "json_model": GeneralResponseModel[per_house_model],
            "regex": regex,
            "label": solution.model_dump_json() if kwargs["solvable"] else "null",
        }
