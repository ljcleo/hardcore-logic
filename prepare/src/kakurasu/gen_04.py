import csv
import random
from itertools import combinations
from tqdm import tqdm
import ast
from functools import lru_cache
import pandas as pd
import numpy as np

class KakurasuGenerator:
    def __init__(self, size):
        self.size = size
        self.all_cells = [(i + 1, j + 1) for i in range(size) for j in range(size)]
        self.available_cells = size * size
        
        # 动态计算黑格数量范围
        self.min_black_cells = max(2, size)  # 最小黑格数
        self.max_black_cells = min(self.available_cells - size, self.available_cells // 2)  # 最大黑格数
        
        # 不可涂黑单元格数量范围
        self.min_forbidden = max(1, size)
        self.max_forbidden = min(2 * size - 2, self.available_cells - self.min_black_cells)

    def compute_sums(self, black_cells):
        """计算行和（该行黑格列号之和）与列和（该列黑格行号之和）"""
        row_sums = [0] * self.size
        col_sums = [0] * self.size
        for (r, c) in black_cells:
            row_sums[r - 1] += c  
            col_sums[c - 1] += r  
        return row_sums, col_sums

    @lru_cache(maxsize=None)
    def generate_row_candidates(self, target_sum):
        """生成单行可能的黑格列组合（列号1-based），其和等于目标行和"""
        candidates = []
        def backtrack(start_col, current_sum, path):
            if current_sum == target_sum:
                candidates.append(tuple(path))
                return
            if current_sum > target_sum:
                return
            for col in range(start_col, self.size + 1):
                path.append(col)
                backtrack(col + 1, current_sum + col, path)
                path.pop()
        backtrack(1, 0, [])
        return candidates

    def validate_solution(self, clue_rows, clue_cols, black_cells, forbidden=None):
        """验证解是否满足线索要求及不可涂黑约束"""
        forbidden = forbidden or set()
        if not black_cells:
            return False
        
        row_sums = [0] * self.size
        col_sums = [0] * self.size
        
        for r, c in black_cells:
            if r < 1 or r > self.size or c < 1 or c > self.size:
                return False
            if (r, c) in forbidden:
                return False
            row_sums[r-1] += c
            col_sums[c-1] += r
        
        return row_sums == clue_rows and col_sums == clue_cols

    def get_all_solutions(self, clue_rows, clue_cols):
        """获取谜题的所有解（无不可涂黑约束时）"""
        row_candidates_list = []
        for r in range(self.size):
            candidates = self.generate_row_candidates(clue_rows[r])
            if not candidates:
                return []
            row_candidates_list.append(candidates)
        all_solutions = []
        def dfs(row_idx, current_col_sums, current_black_cells):
            if row_idx == self.size:
                if current_col_sums == list(clue_cols):
                    all_solutions.append(frozenset(current_black_cells))
                return
            current_row = row_idx + 1
            for candidate in row_candidates_list[row_idx]:
                valid = True
                new_col_sums = current_col_sums.copy()
                new_black_cells = current_black_cells.copy() 
                for col in candidate:
                    col_idx = col - 1
                    new_col_sums[col_idx] += current_row
                    new_black_cells.append((current_row, col))
                    
                    if new_col_sums[col_idx] > clue_cols[col_idx]:
                        valid = False
                        break       
                if valid:
                    dfs(row_idx + 1, new_col_sums, new_black_cells)
        dfs(0, [0] * self.size, [])
        return [set(sol) for sol in all_solutions]

    def select_forbidden_cells(self, original_solution, other_solutions):
        """
        选择不可涂黑的单元格，使得只有原始解有效
        数量控制在 size 到 2*size-2 之间
        """
        diff_cells = set()
        for sol in other_solutions:
            diff_cells.update(sol - original_solution)
        
        if not diff_cells:
            return None 
        num_needed = random.randint(self.min_forbidden, self.max_forbidden)
        all_cells = set(self.all_cells)
        non_original = all_cells - original_solution
        available = list(diff_cells.union(non_original))      
        if len(available) < num_needed:
            return None  
        priority = list(diff_cells)
        random.shuffle(priority)
        remaining = [cell for cell in available if cell not in priority]
        random.shuffle(remaining)
        forbidden = priority[:num_needed]
        if len(forbidden) < num_needed:
            forbidden += remaining[:num_needed - len(forbidden)]
        if original_solution & set(forbidden):
            return None
        return sorted(forbidden)

    def generate_single_puzzle(self):
        """生成满足新约束的谜题：原始多解+加约束后唯一解"""
        for _ in range(200):  
            if self.min_black_cells > self.max_black_cells:
                return None
            n = random.randint(self.min_black_cells, self.max_black_cells)
            original_black = set(random.sample(self.all_cells, n))
            clue_rows, clue_cols = self.compute_sums(original_black)
            all_solutions = self.get_all_solutions(clue_rows, clue_cols)
            if len(all_solutions) < 2:
                continue
            original_set = frozenset(original_black)
            if original_set not in [frozenset(s) for s in all_solutions]:
                continue
            other_solutions = [s for s in all_solutions if frozenset(s) != original_set]
            forbidden = self.select_forbidden_cells(original_black, other_solutions)
            if not forbidden:
                continue
            constrained_solutions = []
            for sol in all_solutions:
                if not (sol & set(forbidden)): 
                    constrained_solutions.append(sol)
            if len(constrained_solutions) == 1 and frozenset(constrained_solutions[0]) == original_set:
                puzzle_str = self._format_puzzle(clue_rows, clue_cols, forbidden)
                solution_bool = self._format_solution(original_black)          
                return {
                    "solution_set": original_black,
                    "clues": [clue_rows, clue_cols],
                    "forbidden": forbidden,
                    "puzzle_str": puzzle_str,
                    "solution_bool": solution_bool
                }
        return None

    def _format_puzzle(self, clue_rows, clue_cols, forbidden):
        """格式化谜题字符串，包含线索和不可涂黑单元格(X)"""
        grid = [['.' for _ in range(self.size)] for _ in range(self.size)]
        for (r, c) in forbidden:
            grid[r-1][c-1] = 'X'
        max_col_clue = max(len(str(c)) for c in clue_cols)
        max_row_clue = max(len(str(r)) for r in clue_rows)
        col_clues_line = ' ' * (max_row_clue + 2) 
        for c in clue_cols:
            col_clues_line += f"{c:>{max_col_clue}} "
        
    
        lines = [col_clues_line]
        for i in range(self.size):
            row_clue = clue_rows[i]
            line = f"{row_clue:>{max_row_clue}}| "
            for j in range(self.size):
                line += f"{grid[i][j]:>{max_col_clue}} "
            lines.append(line)
        
        return '\n'.join(lines)

    def _format_solution(self, black_cells):
        bool_grid = [[False for _ in range(self.size)] for _ in range(self.size)]
        for (r, c) in black_cells:
            bool_grid[r-1][c-1] = True
        lower_bool_grid = []
        for row in bool_grid:
            lower_row = ["true" if cell else "false" for cell in row]
            lower_bool_grid.append(lower_row)
        solution_str = str(lower_bool_grid).replace("'", "").replace('"', "")
        
        return solution_str

    def generate_batch_puzzles(self, count=150, output_path="kakurasu_puzzles.parquet"):
        """批量生成带不可涂黑单元格的谜题并保存为parquet格式"""
        puzzles = []
        seen_clues = set()
        
        print(f"开始生成{count}个 {self.size}x{self.size} 的Kakurasu谜题...")
        generated_count = 0
        attempts = 0
        
        with tqdm(total=count) as pbar:
            while generated_count < count:
                attempts += 1
                if attempts > count * 20:
                    print(f"警告：尝试了{attempts}次，仅生成{generated_count}个谜题，可能需要调整参数")
                    break
                
                puzzle = self.generate_single_puzzle()
                if puzzle:
                    clue_key = (tuple(puzzle["clues"][0]), tuple(puzzle["clues"][1]))
                    if clue_key not in seen_clues:
                        seen_clues.add(clue_key)
                        puzzle_id = f"gen-04--{self.size}x{self.size}-{str(generated_count + 1).zfill(2)}"
                        puzzles.append({
                            "id": puzzle_id,
                            "puzzle": puzzle["puzzle_str"],
                            "solution": puzzle["solution_bool"]
                        })
                        generated_count += 1
                        pbar.update(1)
        df = pd.DataFrame(puzzles)
        df.to_parquet(output_path, engine='pyarrow')
        
        print(f"成功生成{generated_count}个谜题，保存至：{output_path}")
        return df

if __name__ == "__main__":
    size = 7
    generator = KakurasuGenerator(size)
    generator.generate_batch_puzzles(count=50, output_path=r"kakurasupuzzles.parquet")
