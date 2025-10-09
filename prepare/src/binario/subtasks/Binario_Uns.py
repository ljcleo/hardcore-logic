import random
import copy

'''
- task: Binario
- subtask: Unsolvable puzzle
- introduction: Construct a puzzle with only one solution, and then add a value that contradicts the initial solution to an empty cell.
'''

class Binario_Uns:
    def __init__(self, size=6):
        if size % 2 != 0:
            raise ValueError("Grid size must be an even number")
        self.size = size
        self.half = size // 2
        self.board = None

    def generate_valid_board(self):
        self._create_base_template()
        self._shuffle_rows_and_columns()
        return self.board

    def _create_base_template(self):
        self.board = []
        for i in range(self.size):
            if i % 2 == 0:
                row = [0, 1] * self.half
            else:
                row = [1, 0] * self.half
            self.board.append(row)

    def _shuffle_rows_and_columns(self):
        total_swaps = self.size ** 2

        for _ in range(total_swaps):
            if random.choice([True, False]):
                i, j = random.sample(range(self.size), 2)
                self.board[i], self.board[j] = self.board[j], self.board[i]
                if self._check_vertical_triplets():
                    self.board[i], self.board[j] = self.board[j], self.board[i]
            else:
                i, j = random.sample(range(self.size), 2)
                for row in self.board:
                    row[i], row[j] = row[j], row[i]
                if self._check_horizontal_triplets():
                    for row in self.board:
                        row[i], row[j] = row[j], row[i]

    def _check_horizontal_triplets(self):
        for row in self.board:
            for i in range(self.size - 2):
                if row[i] == row[i + 1] == row[i + 2]:
                    return True
        return False

    def _check_vertical_triplets(self):
        for col in range(self.size):
            for i in range(self.size - 2):
                if self.board[i][col] == self.board[i + 1][col] == self.board[i + 2][col]:
                    return True
        return False

    def _has_unique_solution(self, puzzle):
        solution_count = 0
        solutions = []

        def backtrack(row, col):
            nonlocal solution_count
            if solution_count >= 2:
                return

            if row == self.size:
                solution_count += 1
                if solution_count == 1:
                    solutions.append(copy.deepcopy(puzzle))
                return

            next_col = col + 1
            next_row = row
            if next_col == self.size:
                next_col = 0
                next_row += 1

            if puzzle[row][col] is not None:
                backtrack(next_row, next_col)
            else:
                for num in [0, 1]:
                    if self._is_valid_move_puzzle(puzzle, row, col, num):
                        puzzle[row][col] = num
                        backtrack(next_row, next_col)
                        puzzle[row][col] = None

        temp_puzzle = copy.deepcopy(puzzle)
        backtrack(0, 0)
        return solution_count == 1

    def _has_any_solution(self, puzzle):
        solution_count = 0

        def backtrack(row, col):
            nonlocal solution_count
            if solution_count >= 1:
                return

            if row == self.size:
                solution_count += 1
                return

            next_col = col + 1
            next_row = row
            if next_col == self.size:
                next_col = 0
                next_row += 1

            if puzzle[row][col] is not None:
                backtrack(next_row, next_col)
            else:
                for num in [0, 1]:
                    if self._is_valid_move_puzzle(puzzle, row, col, num):
                        puzzle[row][col] = num
                        backtrack(next_row, next_col)
                        puzzle[row][col] = None

        temp_puzzle = copy.deepcopy(puzzle)
        backtrack(0, 0)
        return solution_count > 0

    def _is_valid_move_puzzle(self, puzzle, row, col, num):
        row_vals = [num if j == col else puzzle[row][j] for j in range(self.size)]
        if row_vals.count(num) > self.half:
            return False

        col_vals = [num if i == row else puzzle[i][col] for i in range(self.size)]
        if col_vals.count(num) > self.half:
            return False

        for j in range(self.size - 2):
            trio = [row_vals[j], row_vals[j + 1], row_vals[j + 2]]
            if trio == [num] * 3:
                return False

        for i in range(self.size - 2):
            trio = [col_vals[i], col_vals[i + 1], col_vals[i + 2]]
            if trio == [num] * 3:
                return False

        return True

    def create_puzzle(self):
        solution = self.generate_valid_board()
        puzzle = copy.deepcopy(solution)

        coords = [(i, j) for i in range(self.size) for j in range(self.size)]
        random.shuffle(coords)

        for row, col in coords:
            original = puzzle[row][col]
            puzzle[row][col] = None

            if not self._has_unique_solution(puzzle):
                puzzle[row][col] = original

        return puzzle, solution

    def create_unsolvable_puzzle(self, max_attempts=10):
        for _ in range(max_attempts):
            puzzle, solution = self.create_puzzle()

            empty_cells = [(i, j) for i in range(self.size)
                           for j in range(self.size) if puzzle[i][j] is None]

            if not empty_cells:
                continue

            random.shuffle(empty_cells)
            for row, col in empty_cells:
                original_puzzle = copy.deepcopy(puzzle)

                wrong_value = 1 - solution[row][col]
                puzzle[row][col] = wrong_value

                if not self._has_any_solution(puzzle):
                    # ✅ 直接格式化为 grid 字符串
                    puzzle_str = "\n".join(
                        " ".join(str(cell) if cell is not None else '.' for cell in row)
                        for row in puzzle
                    )
                    return puzzle_str, solution  # 返回字符串化 puzzle + 原始解

                puzzle = original_puzzle

        raise RuntimeError(f"Unable to generate an unsolvable puzzle within {max_attempts} attempts")


if __name__ == '__main__':
    binario_uns = Binario_Uns(6)
    unsolvable_puzzle, original_solution = binario_uns.create_unsolvable_puzzle()
    print(unsolvable_puzzle)
    print(original_solution)
