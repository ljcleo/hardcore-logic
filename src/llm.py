import asyncio
import warnings
from collections.abc import AsyncGenerator, Iterable
from contextlib import asynccontextmanager
from pathlib import Path
from random import shuffle
from typing import Literal, TypedDict

from pydantic import TypeAdapter

from .llm_client import LLM_CLIENTS, LLMClient

type OpenAIMode = Literal["default", "stream", "legacy", "legacy-stream"]
type GeminiMode = Literal["default", "dynamic"]


class APIConfig(TypedDict):
    type: str
    addr: str | None
    ports: list[int] | None
    key: str | list[str]
    proxy: str | None


api_config_adapter: TypeAdapter[dict[str, APIConfig]] = TypeAdapter(dict[str, APIConfig])


class LLM:
    def __init__(
        self,
        api_config_path: Path,
        api_list: Iterable[str],
        api_rep: int,
        timeout: int,
        openai_mode: OpenAIMode,
        gemini_mode: GeminiMode,
    ) -> None:
        with api_config_path.open(mode="rb") as f:
            api_configs: dict[str, APIConfig] = api_config_adapter.validate_json(f.read())

        self._clients: list[LLMClient] = []
        id_buf: list[int] = []

        for api_name in api_list:
            config: APIConfig = api_configs[api_name]
            base_urls: list[str | None] = []
            root_url: str | None = config["addr"]

            if root_url is None:
                base_urls.append(None)
            else:

                def modify_addr(addr: str, port: int | None = None) -> str:
                    if port is not None:
                        addr = f"{addr}:{port}"
                    if "v1" not in addr:
                        addr = f"{addr}/v1"

                    return addr

                if config["ports"] is None or len(config["ports"]) == 0:
                    base_urls.append(modify_addr(root_url))
                else:
                    base_urls.extend(modify_addr(root_url, port=port) for port in config["ports"])

            if openai_mode != "default":
                if config["type"] == "openai":
                    config["type"] = f"openai-{openai_mode}"

                    warnings.warn(
                        f"`openai_mode` set to `{openai_mode}`, using `{config['type']}` as client!"
                    )
                else:
                    warnings.warn(
                        f"`openai_mode` set to `{openai_mode}`, "
                        f"which has no effect on `{config['type']}` clients!"
                    )

            if gemini_mode != "default":
                if config["type"] == "gemini":
                    config["type"] = f"gemini-{gemini_mode}"

                    warnings.warn(
                        f"`gemini_mode` set to `{gemini_mode}`, using `{config['type']}` as client!"
                    )
                else:
                    warnings.warn(
                        f"`gemini_mode` set to `{gemini_mode}`, "
                        f"which has no effect on `{config['type']}` clients!"
                    )

            for key in [config["key"]] if isinstance(config["key"], str) else config["key"]:
                for base_url in base_urls:
                    client_id: int = len(self._clients)

                    self._clients.append(
                        LLM_CLIENTS[config["type"]](
                            api_key=key,
                            base_url=base_url,
                            proxy=config["proxy"],
                            timeout=timeout,
                        )
                    )

                    id_buf.append(client_id)

            print(f"new client `{config['type']}` added")

        shuffle(id_buf)
        id_buf = id_buf * api_rep
        print("client pool size:", len(id_buf))
        self.id_queue: "asyncio.Queue[int]" = asyncio.Queue()

        for client_id in id_buf:
            self.id_queue.put_nowait(client_id)

    @asynccontextmanager
    async def get_client(self) -> AsyncGenerator[LLMClient]:
        client_id: int | None = None

        try:
            client_id = await self.id_queue.get()
            yield self._clients[client_id]
        finally:
            if client_id is not None:
                self.id_queue.put_nowait(client_id)
