import random
import csv
from copy import deepcopy
from gen_01 import SudokuGenerator  #即原来的gen-01中的生成类

class SudokuUnsolvableGenerator:
    def __init__(self, sudoku_generator):
        self.sg = sudoku_generator
        self.size = sudoku_generator.size        
        self.grid_size = sudoku_generator.grid_size  
        self.total_cells = self.size * self.size

    def get_row_col_grid_indices(self, index):
        N = self.size
        G = self.grid_size
        row = index // N
        col = index % N
        grid_row = row // G
        grid_col = col // G
        return row, col, grid_row, grid_col

    def make_unsolvable(self, sudoku_puzzle):
        N = self.size
        G = self.grid_size
        puzzle = deepcopy(sudoku_puzzle)
        zero_indices = [i for i, v in enumerate(puzzle) if v == 0]

        while zero_indices:
            idx = random.choice(zero_indices)
            row, col, grid_row, grid_col = self.get_row_col_grid_indices(idx)
            row_vals = {puzzle[row * N + c] for c in range(N) if puzzle[row * N + c] != 0}
            col_vals = {puzzle[r * N + col] for r in range(N) if puzzle[r * N + col] != 0}
            grid_vals = {puzzle[(grid_row * G + r) * N + (grid_col * G + c)]
                         for r in range(G) for c in range(G)
                         if puzzle[(grid_row * G + r) * N + (grid_col * G + c)] != 0}

            used_vals = row_vals | col_vals | grid_vals
            all_vals = set(range(1, N + 1))
            missing_vals = list(all_vals - used_vals)
            empty_row = [r for r in range(N) if puzzle[row * N + r] == 0 and r != col]
            empty_col = [r for r in range(N) if puzzle[r * N + col] == 0 and r != row]
            empty_grid = [(r, c) for r in range(G) for c in range(G)
                          if puzzle[(grid_row*G + r)*N + (grid_col*G + c)] == 0
                          and (grid_row*G + r != row or grid_col*G + c != col)]

            total_empty = len(empty_row) + len(empty_col) + len(empty_grid)
            if len(missing_vals) <= total_empty:
                for c in empty_row:
                    if not missing_vals: break
                    puzzle[row * N + c] = missing_vals.pop()
                for r in empty_col:
                    if not missing_vals: break
                    puzzle[r * N + col] = missing_vals.pop()
                for r, c in empty_grid:
                    if not missing_vals: break
                    puzzle[(grid_row*G + r)*N + (grid_col*G + c)] = missing_vals.pop()
                return puzzle, (row, col)
            else:
                zero_indices.remove(idx)

        raise Exception("Unable to generate unsolvable Sudoku, please check the input puzzle.")

    def check_unsolvable_position(self, puzzle, index):
        N = self.size
        G = self.grid_size
        if puzzle[index] != 0:
            return False

        row, col, grid_row, grid_col = self.get_row_col_grid_indices(index)
        row_vals = {puzzle[row * N + c] for c in range(N) if puzzle[row * N + c] != 0}
        col_vals = {puzzle[r * N + col] for r in range(N) if puzzle[r * N + col] != 0}
        grid_vals = {puzzle[(grid_row*G + r)*N + (grid_col*G + c)]
                     for r in range(G) for c in range(G)
                     if puzzle[(grid_row*G + r)*N + (grid_col*G + c)] != 0}

        used_vals = row_vals | col_vals | grid_vals
        all_vals = set(range(1, N + 1))
        return used_vals == all_vals


class UnsolvableSudokuCSVGenerator:
    def __init__(self, sudoku_generator, output_path, puzzle_count=50):
        self.sg = sudoku_generator
        self.unsolvable_gen = SudokuUnsolvableGenerator(sudoku_generator)
        self.output_path = output_path
        self.puzzle_count = puzzle_count

    def generate_to_csv(self):
        with open(self.output_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['谜题ID', '终盘数据', '谜题数据', '错误线索位置']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for puzzle_id in range(1, self.puzzle_count + 1):
                sudoku_final = self.sg.sudoku_generate_backtracking()
                sudoku_puzzle = self.sg.sudoku_puzzle_dibble(sudoku_final)
                unsolvable_puzzle, error_pos = self.unsolvable_gen.make_unsolvable(sudoku_puzzle)
                writer.writerow({
                    '谜题ID': puzzle_id,
                    '终盘数据': ','.join(map(str, sudoku_final)),
                    '谜题数据': ','.join(map(str, unsolvable_puzzle)),
                    '错误线索位置': f"{error_pos}"   # 二维坐标
                })


if __name__ == "__main__":
    sg9 = SudokuGenerator(size=9, grid_size=3, puzzle_count=10, max_holes=40)
    csv_path9 = r"unsolvable_sudoku_9x9.csv"
    generator9 = UnsolvableSudokuCSVGenerator(sg9, csv_path9, puzzle_count=10)
    generator9.generate_to_csv()
