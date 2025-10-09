import random
import itertools
import pandas as pd
from pathlib import Path
from collections import defaultdict


class SkyscraperGenerator:
    def __init__(self, size=5, num_puzzles=10, max_attempts=200000):
        self.size = size
        self.num_puzzles = num_puzzles
        self.max_attempts = max_attempts
        self.all_valid_grids = self.generate_all_valid_grids()
        self.clues_count = defaultdict(int)
        for g in self.all_valid_grids:
            k = self._clues_key(self.calculate_clues(g))
            self.clues_count[k] += 1

    def generate_all_valid_grids(self):
        n = self.size
        perms = list(itertools.permutations(range(1, n + 1)))
        results = []

        def backtrack(rows):
            if len(rows) == n:
                results.append([list(r) for r in rows])
                return
            for p in perms:
                conflict = False
                for c in range(n):
                    val = p[c]
                    for existing in rows:
                        if existing[c] == val:
                            conflict = True
                            break
                    if conflict:
                        break
                if conflict:
                    continue
                rows.append(p)
                backtrack(rows)
                rows.pop()

        backtrack([])
        return results
    
    @staticmethod
    def count_visible(buildings):
        max_height = 0
        visible = 0
        for height in buildings:
            if height > max_height:
                visible += 1
                max_height = height
        return visible

    def calculate_clues(self, grid):
        n = self.size
        top = [self.count_visible([grid[r][c] for r in range(n)]) for c in range(n)]
        bottom = [self.count_visible([grid[r][c] for r in reversed(range(n))]) for c in range(n)]
        left = [self.count_visible(row) for row in grid]
        right = [self.count_visible(list(reversed(row))) for row in grid]
        diag_tl_br = self.count_visible([grid[i][i] for i in range(n)])             
        diag_br_tl = self.count_visible([grid[i][i] for i in reversed(range(n))])       
        diag_tr_bl = self.count_visible([grid[i][n - 1 - i] for i in range(n)])      
        diag_bl_tr = self.count_visible([grid[i][n - 1 - i] for i in reversed(range(n))])  
        return [top, bottom, left, right, diag_tl_br, diag_br_tl, diag_tr_bl, diag_bl_tr]

    def _clues_key(self, clues):
        key = []
        for c in clues:
            if isinstance(c, list):
                key.append(tuple(c))
            else:
                key.append(c)
        return tuple(key)

    def generate_random_grid(self):
        base = list(itertools.permutations(range(1, self.size + 1)))
        while True:
            random.shuffle(base)
            grid = []
            for perm in base:
                if not grid:
                    grid.append(list(perm))
                    continue
                if all(perm[c] not in (r[c] for r in grid) for c in range(self.size)):
                    grid.append(list(perm))
                    if len(grid) == self.size:
                        return grid
                    
    def is_unique_solution(self, target_clues):
        k = self._clues_key(target_clues)
        return self.clues_count.get(k, 0) == 1

    def format_puzzle(self, clues):
        n = self.size
        top, bottom, left, right, diag_tl_br, diag_br_tl, diag_tr_bl, diag_bl_tr = clues
        top_mid = "  ".join(map(str, top))
        bottom_mid = "  ".join(map(str, bottom))
        top_line = f" {diag_tl_br}| {top_mid} |{diag_tr_bl}"
        bottom_line = f" {diag_bl_tr}| {bottom_mid} |{diag_br_tl}"
        lines = [top_line]
        dot_row = "  ".join(["."] * n)
        for i in range(n):
            lines.append(f" {left[i]}| {dot_row} |{right[i]}")
        lines.append(bottom_line)
        return "\n".join(lines)
    
    def generate_to_parquet(self, output_path):
        puzzles = []
        attempt = 0
        pid = 1
        while len(puzzles) < self.num_puzzles and attempt < self.max_attempts:
            grid = self.generate_random_grid()
            clues = self.calculate_clues(grid)
            attempt += 1
            if self.is_unique_solution(clues):
                puzzle_str = self.format_puzzle(clues)
                puzzles.append({
                    "id": f"gen-02-diag--{self.size}x{self.size}-{pid:02d}",
                    "puzzle": puzzle_str,
                    "solution": str(grid),
                    "mode": "count"
                })
                pid += 1
        df = pd.DataFrame(puzzles)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)

if __name__ == "__main__": 
    generator = SkyscraperGenerator(size=5, num_puzzles=50, max_attempts=100000) 
    output_file = r"diag-5x5.parquet" 
    generator.generate_to_parquet(output_file)