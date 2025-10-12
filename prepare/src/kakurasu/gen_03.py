import random
from itertools import combinations
from tqdm import tqdm
from functools import lru_cache
import pandas as pd

class KakurasuHiddenClueGenerator:
    def __init__(self, size):
        """
        初始化带隐藏线索的Kakurasu谜题生成器
        :param size: 谜题尺寸（size x size）
        """
        self.size = size
        self.all_cells = [(i + 1, j + 1) for i in range(size) for j in range(size)]  
        self.total_cells = size * size
        self.min_black_cells = max(2, size)  
        self.max_black_cells = min(self.total_cells // 2, self.total_cells - size)  

    @staticmethod
    @lru_cache(maxsize=None)
    def compute_sums(black_cells, size):
        """计算行和（黑格列号之和）与列和（黑格行号之和）"""
        row_sums = [0] * size
        col_sums = [0] * size
        for (r, c) in black_cells:
            row_sums[r - 1] += c
            col_sums[c - 1] += r
        return tuple(row_sums), tuple(col_sums) 

    @staticmethod
    @lru_cache(maxsize=None)
    def generate_row_candidates(num_cols, target_sum):
        """生成某一行满足目标和的候选列集合（1-based列号）"""
        candidates = []
        def backtrack(start_col, current_sum, path):
            if current_sum == target_sum:
                candidates.append(tuple(path))
                return
            if current_sum > target_sum:
                return
            for col in range(start_col, num_cols + 1):
                path.append(col)
                backtrack(col + 1, current_sum + col, path)
                path.pop()
        backtrack(1, 0, [])
        return candidates

    @staticmethod
    @lru_cache(maxsize=None)
    def all_row_combinations(num_cols):
        """返回一行的所有可能黑格组合（用于隐藏线索的行）"""
        cols = range(1, num_cols + 1)
        return [tuple(sub) for k in range(num_cols + 1) for sub in combinations(cols, k)]

    @staticmethod
    def validate_solution(clue_rows, clue_cols, black_cells):
        """验证解是否满足线索要求（-1表示隐藏线索，不校验）"""
        if not black_cells:
            return False
        num_rows = len(clue_rows)
        num_cols = len(clue_cols)
        row_sums = [0] * num_rows
        col_sums = [0] * num_cols

        for r, c in black_cells:
            if r < 1 or r > num_rows or c < 1 or c > num_cols:
                return False
            row_sums[r - 1] += c
            col_sums[c - 1] += r
        for i in range(num_rows):
            if clue_rows[i] != -1 and row_sums[i] != clue_rows[i]:
                return False
        for j in range(num_cols):
            if clue_cols[j] != -1 and col_sums[j] != clue_cols[j]:
                return False
        return True

    def is_unique_solution(self, clue_rows, clue_cols, black_cells):
        """检查完整线索（无-1）的谜题是否唯一解"""
        if not self.validate_solution(clue_rows, clue_cols, black_cells):
            return False
        num_rows = self.size
        num_cols = self.size

        row_candidates = []
        for r in range(num_rows):
            candidates = self.generate_row_candidates(num_cols, clue_rows[r])
            if not candidates:
                return False
            row_candidates.append(candidates)

        solution_count = 0
        def dfs(row_idx, current_col_sums):
            nonlocal solution_count
            if solution_count >= 2: 
                return
            if row_idx == num_rows:
                if list(current_col_sums) == list(clue_cols):
                    solution_count += 1
                return

            current_row = row_idx + 1
            for candidate in row_candidates[row_idx]:
                new_col_sums = list(current_col_sums)
                valid = True
                for col in candidate:
                    col_idx = col - 1
                    new_col_sums[col_idx] += current_row
                    if new_col_sums[col_idx] > clue_cols[col_idx]:
                        valid = False
                        break
                if valid:
                    dfs(row_idx + 1, tuple(new_col_sums)) 

        dfs(0, tuple([0] * num_cols))
        return solution_count == 1

    def is_unique_solution_with_hidden(self, clue_rows, clue_cols, black_cells):
        """检查带隐藏线索（含-1）的谜题是否唯一解"""
        if not self.validate_solution(clue_rows, clue_cols, black_cells):
            return False
        num_rows = self.size
        num_cols = self.size
        row_candidates = []
        for r in range(num_rows):
            if clue_rows[r] == -1:
                candidates = self.all_row_combinations(num_cols)
            else:
                candidates = self.generate_row_candidates(num_cols, clue_rows[r])
            if not candidates:
                return False
            row_candidates.append(candidates)

        solution_count = 0
        def dfs(row_idx, current_col_sums):
            nonlocal solution_count
            if solution_count >= 2:
                return
            if row_idx == num_rows:
                for j in range(num_cols):
                    if clue_cols[j] != -1 and current_col_sums[j] != clue_cols[j]:
                        return
                solution_count += 1
                return

            current_row = row_idx + 1
            for candidate in row_candidates[row_idx]:
                new_col_sums = list(current_col_sums)
                valid = True
                for col in candidate:
                    col_idx = col - 1
                    new_col_sums[col_idx] += current_row
                    if clue_cols[col_idx] != -1 and new_col_sums[col_idx] > clue_cols[col_idx]:
                        valid = False
                        break
                if not valid:
                    continue
                for j in range(num_cols):
                    if clue_cols[j] != -1:
                        max_possible_add = sum(range(current_row + 1, num_rows + 1)) 
                        if new_col_sums[j] > clue_cols[j] or new_col_sums[j] + max_possible_add < clue_cols[j]:
                            valid = False
                            break
                if not valid:
                    continue

                dfs(row_idx + 1, tuple(new_col_sums))  

        dfs(0, tuple([0] * num_cols))
        return solution_count == 1

    def hide_clues_multi(self, original_row_clues, original_col_clues, black_cells, num_to_hide, trials=10):
        """多次尝试隐藏线索，确保隐藏后仍唯一解"""
        for _ in range(trials):
            row_clues = list(original_row_clues)
            col_clues = list(original_col_clues)
            possible_hide_pos = [('row', i) for i in range(self.size)] + [('col', j) for j in range(self.size)]
            random.shuffle(possible_hide_pos)  

            hidden_count = 0
            for pos_type, idx in possible_hide_pos:
                backup = row_clues[idx] if pos_type == 'row' else col_clues[idx]
                if pos_type == 'row':
                    row_clues[idx] = -1
                else:
                    col_clues[idx] = -1
                if self.is_unique_solution_with_hidden(row_clues, col_clues, black_cells):
                    hidden_count += 1
                    if hidden_count == num_to_hide:
                        return tuple(row_clues), tuple(col_clues)  
                else:
                    if pos_type == 'row':
                        row_clues[idx] = backup
                    else:
                        col_clues[idx] = backup
        return None  # 多次尝试失败

    def generate_single_puzzle(self):
        """生成单个完整线索（无隐藏）且唯一解的谜题"""
        for _ in range(200):  
            black_cell_count = random.randint(self.min_black_cells, self.max_black_cells)
            black_cells = tuple(random.sample(self.all_cells, black_cell_count))  
            original_row_clues, original_col_clues = self.compute_sums(black_cells, self.size)
            if self.is_unique_solution(original_row_clues, original_col_clues, black_cells):
                return {
                    "black_cells": black_cells,
                    "original_clues": (original_row_clues, original_col_clues)
                }
        return None 

    def generate_single_puzzle_with_hidden_clues(self, num_to_hide):
        """生成单个带隐藏线索且唯一解的谜题"""
        for _ in range(100):  
            base_puzzle = self.generate_single_puzzle()
            if not base_puzzle:
                continue

            black_cells = base_puzzle["black_cells"]
            original_row, original_col = base_puzzle["original_clues"]
            hidden_clues = self.hide_clues_multi(original_row, original_col, black_cells, num_to_hide)
            if hidden_clues:
                hidden_row, hidden_col = hidden_clues
                puzzle_str = self._format_puzzle(hidden_row, hidden_col)
                solution_str = self._format_solution(black_cells)
                return {
                    "black_cells": black_cells,
                    "original_clues": (original_row, original_col),
                    "hidden_clues": (hidden_row, hidden_col),
                    "puzzle_str": puzzle_str,
                    "solution_str": solution_str
                }
        return None

    def _format_puzzle(self, hidden_row_clues, hidden_col_clues):
        col_clue_strs = [str(c) for c in hidden_col_clues]
        max_col_clue_width = max(len(s) for s in col_clue_strs)
        row_clue_strs = [str(r) for r in hidden_row_clues]
        max_row_clue_width = max(len(s) for s in row_clue_strs)
        col_clue_line = " " * (max_row_clue_width + 2)  
        for c_str in col_clue_strs:
            col_clue_line += f"{c_str:>{max_col_clue_width}} " 
        grid_lines = [col_clue_line]
        for i in range(self.size):
            row_clue = str(hidden_row_clues[i])
            line = f"{row_clue:>{max_row_clue_width}}| "
            line += f" . " * self.size 
            line = line.rstrip()  
            grid_lines.append(line)

        return "\n".join(grid_lines)

    def _format_solution(self, black_cells):
        bool_grid = [[False for _ in range(self.size)] for _ in range(self.size)]
        for (r, c) in black_cells:
            bool_grid[r - 1][c - 1] = True  
        lower_grid = [["true" if cell else "false" for cell in row] for row in bool_grid]
        return str(lower_grid).replace("'", "").replace('"', "")

    def generate_batch_puzzles(self, count=50, output_path="hidden_clue_puzzles.parquet", num_to_hide=None):
        num_to_hide = num_to_hide or self.size
        if num_to_hide < 1 or num_to_hide > 2 * self.size:
            raise ValueError(f"隐藏线索数量需在1~{2*self.size}之间")

        puzzles = []
        seen_clues = set()  
        generated_count = 0

        print(f"开始生成 {count} 个 {self.size}x{self.size} 带隐藏线索的Kakurasu谜题...")
        with tqdm(total=count, desc="生成进度") as pbar:
            while generated_count < count:
                puzzle = self.generate_single_puzzle_with_hidden_clues(num_to_hide)
                if not puzzle:
                    continue

                hidden_clue_key = (puzzle["hidden_clues"][0], puzzle["hidden_clues"][1])
                if hidden_clue_key in seen_clues:
                    continue
                seen_clues.add(hidden_clue_key)

                puzzle_id = f"gen-03--{self.size}x{self.size}-{str(generated_count + 1).zfill(2)}"
                puzzles.append({
                    "id": puzzle_id,
                    "puzzle": puzzle["puzzle_str"],
                    "solution": puzzle["solution_str"]
                })

                generated_count += 1
                pbar.update(1)
        if puzzles:
            df = pd.DataFrame(puzzles)
            df["id"] = df["id"].astype(str)
            df["puzzle"] = df["puzzle"].astype(str)  
            df["solution"] = df["solution"].astype(str) 
            df.to_parquet(output_path, engine="pyarrow", index=False)
            print(f"\n成功生成 {generated_count} 个谜题，保存路径：{output_path}")
            return df
        else:
            print("\n未生成任何谜题，请检查尺寸或隐藏线索数量是否合理")
            return None


if __name__ == "__main__":
    generator = KakurasuHiddenClueGenerator(size=7)
    generator.generate_batch_puzzles(
        count=3,
        output_path=r"hidden_clue_puzzles.parquet",
        num_to_hide=4
    )