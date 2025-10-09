from typing import Any

from .base import Task
from .binario import Binario
from .crypto import Crypto
from .hanoi import Hanoi
from .hitori import Hitori
from .kakurasu import Kakurasu
from .minesweeper import Minesweeper
from .navigation import Navigation
from .skyscraper import Skyscraper
from .sudoku import Sudoku
from .zebra import Zebra

TASKS: dict[str, type[Task[Any]]] = {
    task_cls.name(): task_cls
    for task_cls in (
        Zebra,
        Sudoku,
        Skyscraper,
        Kakurasu,
        Crypto,
        Minesweeper,
        Navigation,
        Binario,
        Hanoi,
        Hitori,
    )
}

__all__ = ["Task", "TASKS"]
