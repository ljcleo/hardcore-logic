import random
import os
import time
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import pyarrow  


class UnsolvableHitoriGenerator:
    def __init__(self, size, max_grid_generations=100000, solve_timeout=30):
        self.size = size
        self.max_grid_generations = max_grid_generations  
        self.solve_timeout = solve_timeout  


    def _generate_raw_grid(self):
        """生成初始随机数字网格（1~size的整数）"""
        return [[random.randint(1, self.size) for _ in range(self.size)] for _ in range(self.size)]

    def _is_valid_black_cell(self, black_cells: set, row: int, col: int) -> bool:
        """检查涂黑单元格是否合法（不与其他黑格相邻）"""
        neighbors = [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]
        for r, c in neighbors:
            if 0 <= r < self.size and 0 <= c < self.size and (r, c) in black_cells:
                return False
        return True

    def _check_row_constraints(self, grid: list, black_cells: set, row: int) -> bool:
        """检查行约束（无重复白格）"""
        seen = set()
        for col in range(self.size):
            if (row, col) in black_cells:
                continue
            val = grid[row][col]
            if val in seen:
                return False
            seen.add(val)
        return True

    def _check_column_constraints(self, grid: list, black_cells: set, col: int) -> bool:
        """检查列约束（无重复白格）"""
        seen = set()
        for row in range(self.size):
            if (row, col) in black_cells:
                continue
            val = grid[row][col]
            if val in seen:
                return False
            seen.add(val)
        return True

    def _is_connected(self, grid: list, black_cells: set) -> bool:
        """检查白格是否连通（BFS实现）"""
        white_cells = set()
        for i in range(self.size):
            for j in range(self.size):
                if (i, j) not in black_cells:
                    white_cells.add((i, j))
        
        if not white_cells:
            return False
        if len(white_cells) == 1:
            return True
        
        start = next(iter(white_cells))
        visited = set([start])
        queue = [start]
        
        while queue:
            r, c = queue.pop(0)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) in white_cells and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        
        return len(visited) == len(white_cells)

    def _solve_hitori(self, grid: list) -> list:
        """求解Hitori谜题，带超时控制，确保完整搜索解空间"""
        solutions = []
        start_time = time.time()

        def backtrack(row: int, col: int, black_cells: set):
            if time.time() - start_time > self.solve_timeout:
                raise TimeoutError("求解超时")
            if len(solutions) >= 1:
                return
            if row == self.size:
                col_valid = all(self._check_column_constraints(grid, black_cells, c) for c in range(self.size))
                if col_valid and self._is_connected(grid, black_cells):
                    solutions.append(frozenset(black_cells))
                return
            next_row, next_col = (row, col + 1) if (col + 1) < self.size else (row + 1, 0)
            current_val = grid[row][col]
            row_conflict = False
            for prev_c in range(col):
                if (row, prev_c) not in black_cells and grid[row][prev_c] == current_val:
                    row_conflict = True
                    break
            if not row_conflict:
                backtrack(next_row, next_col, black_cells.copy())
            if self._is_valid_black_cell(black_cells, row, col):
                new_black = black_cells.copy()
                new_black.add((row, col))
                backtrack(next_row, next_col, new_black)
        try:
            backtrack(0, 0, set())
        except TimeoutError:
            return None
        unique_solutions = list({s for s in solutions})
        return [list(sol) for sol in unique_solutions]

    def _format_puzzle(self, raw_grid: list) -> str:
        """格式化网格为指定字符串格式"""
        formatted_rows = []
        for row in raw_grid:
            formatted_row = f" { '   '.join(map(str, row)) } "
            formatted_rows.append(formatted_row)
        return "\n".join(formatted_rows)

    def generate_single_unsolvable(self) -> list | None:
        """生成单个**真正无解**的谜题（确保每个网格都经过完整验证）"""
        for gen_count in range(self.max_grid_generations):
            raw_grid = self._generate_raw_grid()
            solutions = self._solve_hitori(raw_grid)
            if solutions is not None and len(solutions) == 0:
                return raw_grid
            if (gen_count + 1) % 1000 == 0:
                print(f"已尝试{gen_count + 1}个网格，尚未找到无解谜题...")
        print(f"警告：超过{self.max_grid_generations}次生成尝试，未找到{self.size}x{self.size}无解谜题")
        return None

    def generate_batch_unsolvable(self, num_puzzles: int) -> list:
        """批量生成无解谜题（仅包含真正无解的）"""
        batch_data = []
        generated_count = 0

        with tqdm(total=num_puzzles, desc=f"生成{self.size}x{self.size}无解谜题") as pbar:
            while generated_count < num_puzzles:
                raw_grid = self.generate_single_unsolvable()
                if raw_grid is None:
                    break  
                
                formatted_puzzle = self._format_puzzle(raw_grid)
                puzzle_id = f"gen-04--{self.size}x{self.size}-{generated_count + 1}"
                batch_data.append({
                    "id": puzzle_id,
                    "puzzle": formatted_puzzle,
                    "solution": "null",  
                    "encrypted": False
                })
                generated_count += 1
                pbar.update(1)

        if len(batch_data) < num_puzzles:
            print(f"注意：仅生成{len(batch_data)}/{num_puzzles}个{self.size}x{self.size}无解谜题")
        return batch_data

    def save_to_parquet(self, batch_data: list, output_path: str):
        """保存为Parquet格式"""
        if not batch_data:
            return
        df = pd.DataFrame(batch_data, columns=["id", "puzzle", "solution", "encrypted"])
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_parquet(output_path, engine="pyarrow", index=False)

    def run(self, num_puzzles: int, output_path: str):
        """一键执行生成流程"""
        print(f"开始生成{self.size}x{self.size}无解Hitori谜题（共{num_puzzles}个）")
        print(f"单个网格求解超时时间：{self.solve_timeout}秒，最大生成尝试：{self.max_grid_generations}次")
        batch_data = self.generate_batch_unsolvable(num_puzzles)
        if batch_data:
            self.save_to_parquet(batch_data, output_path)

if __name__ == "__main__":
    generator = UnsolvableHitoriGenerator(
        size=7,
        max_grid_generations=100000,  
        solve_timeout=30 
    )
    output_file = "Hitori_puzzles.parquet"
    generator.run(
        num_puzzles=50,
        output_path=output_file
    )
