import random
import pandas as pd
import pyarrow 

class UnsolvableMinesweeperGenerator:
    def __init__(self, rows=12, cols=12, unknown_percentage_range=(0.7, 0.85), difficulty="easy"):
        self.rows = rows
        self.cols = cols
        self.unknown_percentage_range = unknown_percentage_range
        self.difficulty = difficulty

    def generate_contradictory_minesweeper(self):
        rows, cols = self.rows, self.cols
        min_unknown_percentage, max_unknown_percentage = self.unknown_percentage_range
        if rows == 9 and cols == 9:
            mines_range = (15, 30)
        elif rows == 12 and cols == 12:
            mines_range = (50, 80)
        else:
            mines_range = (max(5, rows * cols // 10), min(30, rows * cols // 3))

        mines_count = random.randint(*mines_range)

        grid = [[0 for _ in range(cols)] for _ in range(rows)]
        mine_positions = []

        cells = [(r, c) for r in range(rows) for c in range(cols)]
        random.shuffle(cells)

        for r, c in cells[:mines_count]:
            grid[r][c] = -1  
            mine_positions.append((r, c))
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] != -1:
                    count = 0
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == -1:
                                count += 1
                    grid[r][c] = count
        non_mine_cells = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] != -1]
        if non_mine_cells:
            r, c = random.choice(non_mine_cells)
            original_value = grid[r][c]
            while True:
                new_value = random.randint(0, 8)
                if new_value != original_value:
                    grid[r][c] = new_value
                    break

        non_mine_cells_count = rows * cols - mines_count
        target_unknown_cells = int(rows * cols * random.uniform(min_unknown_percentage, max_unknown_percentage))
        hidden_non_mines_needed = max(0, target_unknown_cells - mines_count)
        hidden_non_mines_needed = min(hidden_non_mines_needed, non_mine_cells_count)

        non_mines = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] != -1]
        random.shuffle(non_mines)
        hidden_non_mines = set(non_mines[:hidden_non_mines_needed])

        puzzle = []
        for r in range(rows):
            row = []
            for c in range(cols):
                if grid[r][c] == -1 or (r, c) in hidden_non_mines:
                    row.append(-2)  # 未知格子
                else:
                    row.append(grid[r][c])
            puzzle.append(row)

        return puzzle

    def check_contradiction(self, grid):
        """检查是否存在矛盾"""
        rows, cols = len(grid), len(grid[0])
        mines = set()
        safe = set()
        changes_made = True

        while changes_made:
            changes_made = False
            for r in range(rows):
                for c in range(cols):
                    cell = grid[r][c]
                    if 1 <= cell <= 8:
                        unknown, adjacent_mines = [], 0
                        for dr in [-1, 0, 1]:
                            for dc in [-1, 0, 1]:
                                if dr == 0 and dc == 0:
                                    continue
                                nr, nc = r + dr, c + dc
                                if 0 <= nr < rows and 0 <= nc < cols:
                                    if (nr, nc) in mines:
                                        adjacent_mines += 1
                                    elif (nr, nc) not in safe and grid[nr][nc] == -2:
                                        unknown.append((nr, nc))

                        required_mines = cell - adjacent_mines
                        if required_mines < 0 or len(unknown) < required_mines:
                            return True  
                        if adjacent_mines > cell:
                            return True

                        if adjacent_mines == cell:
                            for (nr, nc) in unknown:
                                if (nr, nc) in mines:
                                    return True
                                safe.add((nr, nc))
                                changes_made = True
                        elif len(unknown) == required_mines:
                            for (nr, nc) in unknown:
                                if (nr, nc) in safe:
                                    return True
                                mines.add((nr, nc))
                                changes_made = True
        return False

    def format_puzzle(self, puzzle):
        """把谜题转成字符串格式（. 表示未知，数字直接写）"""
        lines = []
        for row in puzzle:
            line = " ".join("." if cell == -2 else str(cell) for cell in row)
            lines.append(line)
        return "\n".join(lines)

    def generate_dataset(self, n_samples=100, save_path="unsolvable_minesweeper.parquet"):
        data = []
        id_counter = 1
        while len(data) < n_samples:
            puzzle = self.generate_contradictory_minesweeper()
            if self.check_contradiction([row.copy() for row in puzzle]):
                puzzle_str = self.format_puzzle(puzzle)
                entry = {
                    "id": f"unsolvable--{self.difficulty}-{id_counter}",
                    "puzzle": puzzle_str,
                    "solution": "null",
                    "no_adj": False,
                    "letter": False,
                    "regional": False
                }
                data.append(entry)
                id_counter += 1
                print(f"生成 {len(data)} / {n_samples} 个无解谜题")
        df = pd.DataFrame(data)
        df.to_parquet(save_path, engine="pyarrow", index=False)
        print(f"数据已保存到 {save_path}")
        
if __name__ == "__main__":
    gen = UnsolvableMinesweeperGenerator(rows=9, cols=9, difficulty="easy")
    gen.generate_dataset(n_samples=10, save_path="unsolvable_easy.parquet")
