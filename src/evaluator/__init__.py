from .base import Evaluator
from .binario import BinarioEvaluator
from .crypto import CryptoEvaluator
from .hanoi import HanoiEvaluator
from .hitori import HitoriEvaluator
from .kakurasu import KakurasuEvaluator
from .minesweeper import MinesweeperEvaluator
from .navigation import NavigationEvaluator
from .skyscraper import SkyscraperEvaluator
from .sudoku import SudokuEvaluator
from .zebra import ZebraEvaluator

EVALUATORS: dict[str, type[Evaluator]] = {
    evaluator_cls.task(): evaluator_cls
    for evaluator_cls in (
        ZebraEvaluator,
        SudokuEvaluator,
        SkyscraperEvaluator,
        KakurasuEvaluator,
        CryptoEvaluator,
        MinesweeperEvaluator,
        NavigationEvaluator,
        BinarioEvaluator,
        HanoiEvaluator,
        HitoriEvaluator,
    )
}

__all__ = ["Evaluator", "EVALUATORS"]
