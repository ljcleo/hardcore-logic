from typing import Any

from pydantic import BaseModel, TypeAdapter, create_model


def make_model(name: str, **kwargs: type):
    return create_model(
        name,
        __config__=None,
        __doc__=None,
        __base__=None,
        __module__=__name__,
        __validators__=None,
        __cls_kwargs__=None,
        __qualname__=None,
        **kwargs,
    )


def load_json[T](obj_type: type[T], json: str) -> T:
    return TypeAdapter(obj_type).validate_json(json)


def dump_json(obj: Any) -> str:
    if isinstance(obj, BaseModel):
        return obj.model_dump_json()

    return TypeAdapter(type(obj)).dump_json(obj).decode(encoding="utf8")
