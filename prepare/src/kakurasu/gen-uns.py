import random
from itertools import combinations
from tqdm import tqdm
from functools import lru_cache
import pandas as pd

class KakurasuUnsolvableGenerator:
    def __init__(self, size):
        """初始化无解Kakurasu谜题生成器"""
        self.size = size
        self.all_cells = [(i + 1, j + 1) for i in range(size) for j in range(size)]
        self.total_cells = size * size
        self.min_black_cells = max(2, size)
        self.max_black_cells = min(self.total_cells // 2, self.total_cells - size)

    @staticmethod
    @lru_cache(maxsize=None)
    def compute_sums(black_cells, size):
        """计算行和与列和"""
        row_sums = [0] * size
        col_sums = [0] * size
        for (r, c) in black_cells:
            row_sums[r - 1] += c
            col_sums[c - 1] += r
        return tuple(row_sums), tuple(col_sums)

    @staticmethod
    @lru_cache(maxsize=None)
    def generate_row_candidates(num_cols, target_sum):
        """生成单行满足目标和的候选列集合"""
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
    def validate_solution(clue_rows, clue_cols, black_cells):
        """验证解是否满足线索要求"""
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

        return row_sums == list(clue_rows) and col_sums == list(clue_cols)

    def has_solution(self, clue_rows, clue_cols):
        """判断给定线索是否有解"""
        num_rows = self.size
        num_cols = self.size

        row_candidates = []
        for r in range(num_rows):
            candidates = self.generate_row_candidates(num_cols, clue_rows[r])
            if not candidates:
                return False
            row_candidates.append(candidates)

        def dfs(row_idx, current_col_sums):
            if row_idx == num_rows:
                return current_col_sums == list(clue_cols)

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

                if valid and dfs(row_idx + 1, tuple(new_col_sums)):
                    return True
            return False

        return dfs(0, tuple([0] * num_cols))

    def is_unique_solution(self, clue_rows, clue_cols, black_cells):
        """检查给定谜题是否有唯一解"""
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
        
        def dfs(row_idx, current_col_sums, current_black_cells):
            nonlocal solution_count
            if solution_count >= 2:
                return
            
            if row_idx == num_rows:
                if current_col_sums == list(clue_cols):
                    solution_count += 1
                return
            
            current_row = row_idx + 1
            for candidate in row_candidates[row_idx]:
                new_col_sums = list(current_col_sums)
                new_black_cells = current_black_cells.copy()
                valid = True
                
                for col in candidate:
                    col_idx = col - 1
                    new_col_sums[col_idx] += current_row
                    new_black_cells.append((current_row, col))
                    
                    if new_col_sums[col_idx] > clue_cols[col_idx]:
                        valid = False
                        break
                
                if valid:
                    dfs(row_idx + 1, new_col_sums, new_black_cells)
                    if solution_count >= 2:
                        return
        
        dfs(0, [0] * num_cols, [])
        return solution_count == 1

    def _format_puzzle(self, clue_rows, clue_cols):
        """格式化puzzle字符串"""
        col_clue_strs = [str(c) for c in clue_cols]
        max_col_clue_width = max(len(s) for s in col_clue_strs) if col_clue_strs else 2
        row_clue_strs = [str(r) for r in clue_rows]
        max_row_clue_width = max(len(s) for s in row_clue_strs) if row_clue_strs else 2

        col_clue_line = " " * (max_row_clue_width + 2)
        for c in clue_cols:
            col_clue_line += f"{c:>{max_col_clue_width}} "

        grid_lines = [col_clue_line]
        for row_idx in range(self.size):
            row_clue = clue_rows[row_idx]
            line = f"{row_clue:>{max_row_clue_width}}| "
            line += f" . " * self.size
            line = line.rstrip()
            grid_lines.append(line)

        return "\n".join(grid_lines)

    def generate_single_unsolvable_puzzle(self):
        """生成单个无解谜题"""
        for _ in range(100):
            if self.min_black_cells > self.max_black_cells:
                return None
            black_cell_count = random.randint(self.min_black_cells, self.max_black_cells)
            black_cells = tuple(random.sample(self.all_cells, black_cell_count))
            original_row, original_col = self.compute_sums(black_cells, self.size)

            clue_rows = list(original_row)
            clue_cols = list(original_col)
            if self.size >= 2:
                swap_type = random.choice(["row", "col"])
                if swap_type == "row":
                    i, j = random.sample(range(self.size), 2)
                    clue_rows[i], clue_rows[j] = clue_rows[j], clue_rows[i]
                else:
                    i, j = random.sample(range(self.size), 2)
                    clue_cols[i], clue_cols[j] = clue_cols[j], clue_cols[i]

            if not self.has_solution(tuple(clue_rows), tuple(clue_cols)):
                puzzle_str = self._format_puzzle(clue_rows, clue_cols)
                return {
                    "clues": (clue_rows, clue_cols),
                    "puzzle_str": puzzle_str
                }
        return None

    def generate_batch_unsolvable_puzzles(self, count=40, output_path="unsolvable_puzzles.parquet"):
        """批量生成无解谜题并保存为Parquet（确保solution为"null"字符串）"""
        puzzles = []
        seen_clues = set()
        generated_count = 0

        print(f"开始生成 {count} 个无解 {self.size}x{self.size} Kakurasu谜题...")
        with tqdm(total=count, desc="生成进度") as pbar:
            while generated_count < count:
                single_puzzle = self.generate_single_unsolvable_puzzle()
                if not single_puzzle:
                    continue

                clue_key = (tuple(single_puzzle["clues"][0]), tuple(single_puzzle["clues"][1]))
                if clue_key in seen_clues:
                    continue
                seen_clues.add(clue_key)

                puzzle_id = f"unsolvable--{self.size}x{self.size}-{str(generated_count + 1).zfill(2)}"
                puzzles.append({
                    "id": puzzle_id,
                    "puzzle": single_puzzle["puzzle_str"],
                    "solution": "null"  
                })

                generated_count += 1
                pbar.update(1)

        if puzzles:
            df = pd.DataFrame(puzzles)
            df["id"] = df["id"].astype(str)
            df["puzzle"] = df["puzzle"].astype(str)
            df["solution"] = df["solution"].astype(str)  
            df.to_parquet(output_path, engine="pyarrow", index=False)
            print(f"\n成功生成 {generated_count} 个无解谜题，保存路径：{output_path}")
            return df
        else:
            print("\n未生成任何无解谜题，请检查尺寸是否合理")
            return None


if __name__ == "__main__":
    size = 7
    generator = KakurasuUnsolvableGenerator(size=size)
    generator.generate_batch_unsolvable_puzzles(
        count=5,
        output_path=r"unsolvable.parquet"
    )
