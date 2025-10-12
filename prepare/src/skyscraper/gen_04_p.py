import csv
import random
import os
import re
from itertools import chain
from typing import List, Optional, Tuple
import pandas as pd

class SkyscraperPuzzleGenerator:
    def __init__(self, puzzle_mode: str = "count"):
        if puzzle_mode not in ["count", "sum"]:
            raise ValueError("Puzzle mode only supports 'count' or 'sum'")
        self.mode = puzzle_mode
        self.parquet_engine = "pyarrow"
        self.default_save_path = (
            r"puzzles.parquet"
            .format(mode=self.mode)
        )

    @staticmethod
    def generate_skyscraper_grid(size: int) -> List[List[int]]:
        grid = [[0 for _ in range(size)] for _ in range(size)]
        first_row = list(range(1, size + 1))
        random.shuffle(first_row)
        grid[0] = first_row
        for row in range(1, size):
            used_in_row = set()
            for col in range(size):
                possible_nums = []
                for num in range(1, size + 1):
                    if num in used_in_row:
                        continue
                    col_duplicate = any(grid[r][col] == num for r in range(row))
                    if not col_duplicate:
                        possible_nums.append(num)
                if not possible_nums:
                    return SkyscraperPuzzleGenerator.generate_skyscraper_grid(size)
                chosen_num = random.choice(possible_nums)
                grid[row][col] = chosen_num
                used_in_row.add(chosen_num)
        return grid

    def _calculate_visibility(self, grid: List[List[int]], size: int) -> List[List[int]]:
        if self.mode == "count":
            return self._calc_visibility_count(grid, size)
        else:
            return self._calc_visibility_sum(grid, size)

    @staticmethod
    def _calc_visibility_count(grid: List[List[int]], size: int) -> List[List[int]]:
        top, bottom, left, right = [], [], [], []
        for col in range(size):
            max_h, count = 0, 0
            for row in range(size):
                if grid[row][col] > max_h:
                    max_h = grid[row][col]
                    count += 1
            top.append(count)
        for col in range(size):
            max_h, count = 0, 0
            for row in range(size-1, -1, -1):
                if grid[row][col] > max_h:
                    max_h = grid[row][col]
                    count += 1
            bottom.append(count)
        for row in range(size):
            max_h, count = 0, 0
            for col in range(size):
                if grid[row][col] > max_h:
                    max_h = grid[row][col]
                    count += 1
            left.append(count)
        for row in range(size):
            max_h, count = 0, 0
            for col in range(size-1, -1, -1):
                if grid[row][col] > max_h:
                    max_h = grid[row][col]
                    count += 1
            right.append(count)
        return [top, bottom, left, right]

    @staticmethod
    def _calc_visibility_sum(grid: List[List[int]], size: int) -> List[List[int]]:
        top, bottom, left, right = [], [], [], []
        for col in range(size):
            max_h, sum_h = 0, 0
            for row in range(size):
                if grid[row][col] > max_h:
                    sum_h += grid[row][col]
                    max_h = grid[row][col]
            top.append(sum_h)
        for col in range(size):
            max_h, sum_h = 0, 0
            for row in range(size-1, -1, -1):
                if grid[row][col] > max_h:
                    sum_h += grid[row][col]
                    max_h = grid[row][col]
            bottom.append(sum_h)
        for row in range(size):
            max_h, sum_h = 0, 0
            for col in range(size):
                if grid[row][col] > max_h:
                    sum_h += grid[row][col]
                    max_h = grid[row][col]
            left.append(sum_h)
        for row in range(size):
            max_h, sum_h = 0, 0
            for col in range(size-1, -1, -1):
                if grid[row][col] > max_h:
                    sum_h += grid[row][col]
                    max_h = grid[row][col]
            right.append(sum_h)
        return [top, bottom, left, right]

    @staticmethod
    def hide_random_clues(visibility: List[List[int]], size: int) -> List[List[int]]:
        min_hide = size
        max_hide = size + 2
        hide_count = random.randint(min_hide, max_hide)
        flat_clues = list(chain.from_iterable(visibility))
        total_clues = len(flat_clues)
        hide_count = min(hide_count, total_clues)
        hide_indices = random.sample(range(total_clues), hide_count)
        for idx in hide_indices:
            flat_clues[idx] = -1
        hidden_visibility = []
        idx = 0
        for dir_clues in visibility:
            dir_len = len(dir_clues)
            hidden_visibility.append(flat_clues[idx:idx+dir_len])
            idx += dir_len
        return hidden_visibility

    @staticmethod
    def hide_random_clues(visibility: List[List[int]], size: int) -> List[List[int]]:
        min_hide = size
        max_hide = size + 2
        hide_count = random.randint(min_hide, max_hide)
        flat_clues = list(chain.from_iterable(visibility))
        total_clues = len(flat_clues)
        hide_count = min(hide_count, total_clues)
        hide_indices = random.sample(range(total_clues), hide_count)
        for idx in hide_indices:
            flat_clues[idx] = -1
        hidden_visibility = []
        idx = 0
        for dir_clues in visibility:
            dir_len = len(dir_clues)
            hidden_visibility.append(flat_clues[idx:idx+dir_len])
            idx += dir_len
        return hidden_visibility

    def _format_puzzle_string(self, hidden_clues: List[List[int]], size: int) -> str:
        top = hidden_clues[0]    
        bottom = hidden_clues[1] 
        left = hidden_clues[2]   
        right = hidden_clues[3]  
        top_clues_formatted = [f"{c:>2}" for c in top]
        top_line = f"-1|{' '.join(top_clues_formatted)}| -1"
        dots_raw = " . " * size
        dots_formatted = dots_raw.rstrip()
        middle_lines = []
        for row_idx in range(size):
            left_val = f"{left[row_idx]:>2}"
            right_val = f"{right[row_idx]:>2}"
            middle_line = f"{left_val}|{dots_formatted}|{right_val}"
            middle_lines.append(middle_line)
        bottom_clues_formatted = [f"{c:>2}" for c in bottom]
        bottom_line = f"-1|{' '.join(bottom_clues_formatted)}| -1"
        return "\n".join([top_line] + middle_lines + [bottom_line])

    @staticmethod
    def _format_solution(grid: List[List[int]]) -> str:
        return str(grid).replace(" ", "")

    def _get_max_original_id(self, save_path: str) -> int:
        if not os.path.exists(save_path):
            return 0
        try:
            df_exist = pd.read_parquet(save_path, engine=self.parquet_engine)
            if "id" not in df_exist.columns:
                return 0
            id_pattern = re.compile(r"gen-04-p-(count|sum)--\d+x\d+-(\d+)")
            max_id = 0
            for puzzle_id in df_exist["id"]:
                match = id_pattern.match(puzzle_id)
                if match:
                    original_id = int(match.group(2))
                    if original_id > max_id:
                        max_id = original_id
            return max_id
        except Exception as e:
            return 0

    def generate_and_save(
        self,
        sizes: List[int],
        count_per_size: int,
        save_path: Optional[str] = None
    ) -> None:
        final_path = save_path if save_path else self.default_save_path
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        max_original_id = self._get_max_original_id(final_path)
        puzzle_data = []
        for size in sizes:
            for _ in range(count_per_size):
                max_original_id += 1
                grid = self.generate_skyscraper_grid(size)
                full_clues = self._calculate_visibility(grid, size)
                hidden_clues = self.hide_random_clues(full_clues, size)
                puzzle_id = (
                    f"gen-04-p-{self.mode}--{size}x{size}-{max_original_id:02d}"
                )
                puzzle_str = self._format_puzzle_string(hidden_clues, size)
                solution_str = self._format_solution(grid)
                puzzle_data.append({
                    "id": puzzle_id,
                    "puzzle": puzzle_str,
                    "solution": solution_str,
                    "mode": self.mode
                })
        df_new = pd.DataFrame(puzzle_data)
        if os.path.exists(final_path):
            df_exist = pd.read_parquet(final_path, engine=self.parquet_engine)
            df_combined = pd.concat([df_exist, df_new], ignore_index=True)
            df_combined.to_parquet(final_path, engine=self.parquet_engine, index=False)
        else:
            df_new.to_parquet(final_path, engine=self.parquet_engine, index=False)


if __name__ == "__main__":
    TARGET_MODES = ["count", "sum"]  
    PUZZLE_SIZES = [6, 7, 8]      
    COUNT_PER_SIZE = 50            
    CUSTOM_SAVE_PATH = r"puzzle.parquet"      
    for mode in TARGET_MODES:
        generator = SkyscraperPuzzleGenerator(puzzle_mode=mode)
        generator.generate_and_save(
            sizes=PUZZLE_SIZES,
            count_per_size=COUNT_PER_SIZE,
            save_path=CUSTOM_SAVE_PATH
        )