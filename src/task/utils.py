from collections.abc import Iterable, Mapping
from typing import Any


def make_choice_regex(inner: Iterable[str], *, is_str: bool) -> str:
    regex: str = f"({'|'.join(inner)})"
    return f'"{regex}"' if is_str else regex


def make_list_regex(inner: Iterable[str]) -> str:
    return rf"\[{','.join(inner)}\]"


def make_dict_regex(inner: Mapping[str, str]) -> str:
    return rf"\{{{','.join(f'"{k}":{v}' for k, v in inner.items())}\}}"


def make_general_regex(inner: str) -> str:
    return rf'\{{"solvable":(true|false),"solution":(null|{inner})\}}'


def replace_value_in_dict(item: Any, original_schema: dict[str, Any]) -> Any:
    if isinstance(item, list):
        return [replace_value_in_dict(i, original_schema) for i in item]
    elif isinstance(item, dict):
        if list(item.keys()) == ["$ref"]:
            definitions: list[str] = item["$ref"][2:].split("/")
            res: dict[str, Any] = original_schema.copy()

            for definition in definitions:
                res = res[definition]

            return res
        else:
            return {key: replace_value_in_dict(i, original_schema) for key, i in item.items()}
    else:
        return item
