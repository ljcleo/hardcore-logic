import json
from pydantic import TypeAdapter


def load_json[T](obj_type: type[T], doc: str) -> T:
    return TypeAdapter(obj_type).validate_json(doc)


def compress_json(doc: str) -> str:
    return json.dumps(json.loads(doc), ensure_ascii=False, separators=(",", ":"))
