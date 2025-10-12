import itertools
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import os


class SkyscraperUnsolvableGenerator:

    def __init__(self, size=5, count=50, save_path="unsolvable_skyscraper.parquet"):
        self.size = size
        self.count = count
        self.save_path = save_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.valid_grids = self._generate_all_valid_grids(size)

    def _generate_all_valid_grids(self, size):
        all_grids = []
        if size == 4:
            for row1 in itertools.permutations(range(1, size + 1)):
                for row2 in itertools.permutations(range(1, size + 1)):
                    if any(row2[i] == row1[i] for i in range(size)):
                        continue
                    for row3 in itertools.permutations(range(1, size + 1)):
                        if any(row3[i] in (row1[i], row2[i]) for i in range(size)):
                            continue
                        row4 = []
                        for col in range(size):
                            col_values = {row1[col], row2[col], row3[col]}
                            row4.append(({1, 2, 3, 4} - col_values).pop())
                        if len(set(row4)) == size:
                            all_grids.append([list(row1), list(row2), list(row3), list(row4)])
        elif size == 5:
            for row1 in itertools.permutations(range(1, size + 1)):
                for row2 in itertools.permutations(range(1, size + 1)):
                    if any(row2[i] == row1[i] for i in range(size)):
                        continue
                    for row3 in itertools.permutations(range(1, size + 1)):
                        if any(row3[i] in (row1[i], row2[i]) for i in range(size)):
                            continue
                        for row4 in itertools.permutations(range(1, size + 1)):
                            if any(row4[i] in (row1[i], row2[i], row3[i]) for i in range(size)):
                                continue
                            row5 = []
                            for col in range(size):
                                col_values = {row1[col], row2[col], row3[col], row4[col]}
                                row5.append((set(range(1, size + 1)) - col_values).pop())
                            if len(set(row5)) == size:
                                all_grids.append([list(row1), list(row2), list(row3), list(row4), list(row5)])
        return all_grids

    def _compute_clues(self, grid):
        size = len(grid)
        top, bottom, left, right = [0]*size, [0]*size, [0]*size, [0]*size
        for col in range(size):
            max_height = visible = 0
            for row in range(size):
                if grid[row][col] > max_height:
                    max_height = grid[row][col]
                    visible += 1
            top[col] = visible
        for col in range(size):
            max_height = visible = 0
            for row in range(size-1, -1, -1):
                if grid[row][col] > max_height:
                    max_height = grid[row][col]
                    visible += 1
            bottom[col] = visible
        for row in range(size):
            max_height = visible = 0
            for col in range(size):
                if grid[row][col] > max_height:
                    max_height = grid[row][col]
                    visible += 1
            left[row] = visible
        for row in range(size):
            max_height = visible = 0
            for col in range(size-1, -1, -1):
                if grid[row][col] > max_height:
                    max_height = grid[row][col]
                    visible += 1
            right[row] = visible
        return top, bottom, left, right

    def _has_solution(self, clues):
        top, bottom, left, right = clues
        for grid in self.valid_grids:
            gt, gb, gl, gr = self._compute_clues(grid)
            if gt == top and gb == bottom and gl == left and gr == right:
                return True
        return False

    def _generate_random_clues(self):
        size = self.size
        top = [random.randint(1, size) for _ in range(size)]
        bottom = [random.randint(1, size) for _ in range(size)]
        left = [random.randint(1, size) for _ in range(size)]
        right = [random.randint(1, size) for _ in range(size)]
        return top, bottom, left, right

    def generate_unsolvable_puzzles(self):
        puzzles = []
        with tqdm(total=self.count, desc=f"Generate{self.size}x{self.size} unsolvable puzzles") as pbar:
            while len(puzzles) < self.count:
                clues = self._generate_random_clues()
                if not self._has_solution(clues):
                    puzzles.append(clues)
                    pbar.update(1)
        return puzzles

    def _format_puzzle(self, clues):
        top, bottom, left, right = clues
        size = self.size
        lines = [f"-1| {'  '.join(map(str, top))}|-1"]
        for i in range(size):
            row = f"{left[i]}| {'  '.join(['.'] * size)}| {right[i]}"
            lines.append(row)
        lines.append(f"-1| {'  '.join(map(str, bottom))}|-1")
        return "\n".join(lines)

    def save_to_parquet(self):
        puzzles = self.generate_unsolvable_puzzles()
        rows = []
        for idx, clues in enumerate(puzzles, 1):
            count_str = f"{idx:03d}"
            puzzle_str = self._format_puzzle(clues)
            rows.append({
                "id": f"unsolvable--{self.size}x{self.size}-{count_str}",
                "puzzle": puzzle_str,
                "solution": "null",
                "mode": "count"
            })

        df = pd.DataFrame(rows)
        df.to_parquet(self.save_path, index=False)

if __name__ == "__main__":
    generator = SkyscraperUnsolvableGenerator(size=5, count=10,
        save_path=r"unsolvable_5x5.parquet")
    generator.save_to_parquet()