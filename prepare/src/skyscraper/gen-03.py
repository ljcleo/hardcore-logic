import random
import itertools
import pandas as pd
from pathlib import Path


class PartialSkyscraperPuzzleGenerator:
    def __init__(self, size=5, mode="count"):
        assert mode in ("count", "sum"), "mode必须是'count'或'sum'"
        self.size = size
        self.mode = mode
        self.all_valid_grids = self.generate_all_valid_grids()

    def generate_skyscraper_grid(self):
        base = list(itertools.permutations(range(1, self.size + 1)))
        random.shuffle(base)
        grid = []
        for perm in base:
            if not grid:
                grid.append(list(perm))
                continue
            valid = True
            for col in range(self.size):
                if perm[col] in [row[col] for row in grid]:
                    valid = False
                    break
            if valid:
                grid.append(list(perm))
                if len(grid) == self.size:
                    break
        return grid

    def generate_all_valid_grids(self):
        all_grids = []
        for row1 in itertools.permutations(range(1, self.size + 1)):
            for row2 in itertools.permutations(range(1, self.size + 1)):
                if any(row2[i] == row1[i] for i in range(self.size)):
                    continue
                for row3 in itertools.permutations(range(1, self.size + 1)):
                    if any(row3[i] in (row1[i], row2[i]) for i in range(self.size)):
                        continue
                    for row4 in itertools.permutations(range(1, self.size + 1)):
                        if any(row4[i] in (row1[i], row2[i], row3[i]) for i in range(self.size)):
                            continue
                        row5 = []
                        valid_row5 = True
                        for col in range(self.size):
                            used = {row1[col], row2[col], row3[col], row4[col]}
                            available = set(range(1, self.size + 1)) - used
                            if len(available) != 1:
                                valid_row5 = False
                                break
                            row5.append(available.pop())
                        if valid_row5 and len(set(row5)) == self.size:
                            all_grids.append([
                                list(row1), list(row2), list(row3), list(row4), row5
                            ])
        return all_grids

    def count_visible(self, buildings):
        max_height = 0
        visible = 0
        for height in buildings:
            if height > max_height:
                visible += 1
                max_height = height
        return visible
    def sum_visible(self, buildings):
        max_height = 0
        visible_sum = 0
        for height in buildings:
            if height > max_height:
                visible_sum += height
                max_height = height
        return visible_sum
    def calculate_clues(self, grid):
        func = self.count_visible if self.mode == "count" else self.sum_visible
        size = len(grid)
        top = [func([grid[r][c] for r in range(size)]) for c in range(size)]
        bottom = [func([grid[r][c] for r in reversed(range(size))]) for c in range(size)]
        left = [func(row) for row in grid]
        right = [func(list(reversed(row))) for row in grid]
        return [top, bottom, left, right]

    def is_unique_solution_with_hidden(self, target_clues):
        match_count = 0
        for grid in self.all_valid_grids:
            grid_clues = self.calculate_clues(grid)
            match = True
            for i in range(4):
                for j in range(self.size):
                    if target_clues[i][j] != 0 and grid_clues[i][j] != target_clues[i][j]:
                        match = False
                        break
                if not match:
                    break
            if match:
                match_count += 1
                if match_count > 1:
                    return False
        return match_count == 1

    def hide_specific_clues(self, original_clues, min_hide=8, max_hide=12):
        #In sum mode, min_ide is 5 and max_ide is 8.
        total_positions = [(i, j) for i in range(4) for j in range(self.size)]
        target_hide = random.randint(min_hide, max_hide)
        for _ in range(1000):
            temp = [row.copy() for row in original_clues]
            to_hide = random.sample(total_positions, target_hide)
            for i, j in to_hide:
                temp[i][j] = 0
            if self.is_unique_solution_with_hidden(temp):
                return temp
        return original_clues

    def format_puzzle_display(self, clues):
        size = self.size
        top, bottom, left, right = clues
        top = [-1 if v == 0 else v for v in top]
        bottom = [-1 if v == 0 else v for v in bottom]
        left = [-1 if v == 0 else v for v in left]
        right = [-1 if v == 0 else v for v in right]
        lines = []
        corner = "-1"
        lines.append(f"{corner}| {' '.join(f'{v:2d}' for v in top)} |{corner}")
        for i in range(size):
            lines.append(f"{left[i]:2d}| {' '.join([' .']*size)} |{right[i]:2d}")
        lines.append(f"{corner}| {' '.join(f'{v:2d}' for v in bottom)} |{corner}")
        return "\n".join(lines)

    def generate_puzzles(self, output_path, num_puzzles=50, max_attempts=100000):
        data = []
        puzzle_id = 1
        attempt = 0
        while len(data) < num_puzzles and attempt < max_attempts:
            attempt += 1
            solution_grid = self.generate_skyscraper_grid()
            full_clues = self.calculate_clues(solution_grid)
            if not self.is_unique_solution_with_hidden(full_clues):
                continue
            hidden_clues = self.hide_specific_clues(full_clues)
            puzzle_str = self.format_puzzle_display(hidden_clues)
            pid = f"gen-03-{self.mode}--5x5-{puzzle_id:03d}"
            data.append({
                "id": pid,
                "puzzle": puzzle_str,
                "solution": str(solution_grid),
                "mode": self.mode
            })
            puzzle_id += 1
        df = pd.DataFrame(data)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)

if __name__ == "__main__":
    gen_count = PartialSkyscraperPuzzleGenerator(mode="count")
    gen_count.generate_puzzles(
        output_path=r"gen-03-count--5x5.parquet",
        num_puzzles=50
    )

