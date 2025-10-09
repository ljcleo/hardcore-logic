import copy
import random
import csv
import os

class SudokuExtremalGenerator:
    def __init__(self, num_puzzles: int, csv_path: str):
        self.num_puzzles = num_puzzles
        self.csv_path = csv_path
        self.fieldnames = [
            "谜题ID", "数独类型", "终盘数据", 
            "谜题数据", "挖空数",   
        ]

    def find_empty(self, board: list[list[int]]) -> tuple[int, int] | None:
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] == 0:
                    return (i, j)
        return None

    def is_valid_normal(self, board: list[list[int]], num: int, pos: tuple[int, int]) -> bool:
        """检查在pos位置放置num是否满足普通约束"""
        for i in range(len(board[0])):
            if board[pos[0]][i] == num and pos[1] != i:
                return False
        for i in range(len(board)):
            if board[i][pos[1]] == num and pos[0] != i:
                return False
        box_x = pos[1] // 3
        box_y = pos[0] // 3
        for i in range(box_y*3, box_y*3 + 3):
            for j in range(box_x * 3, box_x*3 + 3):
                if board[i][j] == num and (i, j) != pos:
                    return False
        return True

    def calculate_subgrid_index_value(self, subgrid: list[list[int]]) -> int:
        """计算3x3子网格的index value"""
        index_matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]  
        return sum(index_matrix[i][j] * subgrid[i][j] for i in range(3) for j in range(3))

    def get_subgrid(self, board: list[list[int]], box_row: int, box_col: int) -> list[list[int]]:
        start_row = box_row * 3
        start_col = box_col * 3
        return [[board[start_row+i][start_col+j] for j in range(3)] for i in range(3)]

    def find_max_index_value_subgrid_pos(self, board: list[list[int]]) -> tuple[int, int]:
        """找到终盘中index value最大的子网格位置"""
        max_value = -1
        max_pos = (0, 0)
        for box_row in range(3):
            for box_col in range(3):
                subgrid = self.get_subgrid(board, box_row, box_col)
                current_val = self.calculate_subgrid_index_value(subgrid)
                if current_val > max_value:
                    max_value = current_val
                    max_pos = (box_row, box_col)
        return max_pos

    def is_solution_meet_extremal_constraint(
        self, full_solution: list[list[int]], target_box_pos: tuple[int, int]
    ) -> bool:
        """检查终解是否满足极值约束"""
        target_subgrid = self.get_subgrid(full_solution, target_box_pos[0], target_box_pos[1])
        target_val = self.calculate_subgrid_index_value(target_subgrid)
        for box_row in range(3):
            for box_col in range(3):
                if (box_row, box_col) == target_box_pos:
                    continue
                current_subgrid = self.get_subgrid(full_solution, box_row, box_col)
                current_val = self.calculate_subgrid_index_value(current_subgrid)
                if current_val > target_val:
                    return False
        return True

    def solve_all_sudoku(
        self, board: list[list[int]], solutions: list[list[list[int]]],
        constraint_type: str = "normal", target_box_pos: tuple[int, int] | None = None
    ) -> None:
        """求解所有有效解，按约束类型筛选"""
        find = self.find_empty(board)
        if not find:  # 找到完整解
            if constraint_type == "normal":
                solutions.append(copy.deepcopy(board))
            else: 
                if self.is_solution_meet_extremal_constraint(board, target_box_pos):
                    solutions.append(copy.deepcopy(board))
            return
        else:
            row, col = find
        for num in range(1, 10):
            if self.is_valid_normal(board, num, (row, col)):
                board[row][col] = num
                self.solve_all_sudoku(board, solutions, constraint_type, target_box_pos)
                board[row][col] = 0  # 回溯

    def limit_solve(
        self, board: list[list[int]], solutions: list[list[list[int]]],
        max_solutions: int = 2, constraint_type: str = "normal",
        target_box_pos: tuple[int, int] | None = None
    ) -> None:
        if len(solutions) >= max_solutions:
            return
        
        find = self.find_empty(board)
        if not find:  
            if constraint_type == "normal":
                solutions.append(copy.deepcopy(board))
            else:
                if self.is_solution_meet_extremal_constraint(board, target_box_pos):
                    solutions.append(copy.deepcopy(board))
            return
        else:
            row, col = find
        for num in range(1, 10):
            if self.is_valid_normal(board, num, (row, col)):
                board[row][col] = num
                self.limit_solve(board, solutions, max_solutions, constraint_type, target_box_pos)
                board[row][col] = 0
                if len(solutions) >= max_solutions:
                    return

    def check_sudoku_solutions(
        self, board: list[list[int]], constraint_type: str = "normal",
        target_box_pos: tuple[int, int] | None = None
    ) -> int:
        board_copy = copy.deepcopy(board)
        solutions = []
        self.limit_solve(board_copy, solutions, 2, constraint_type, target_box_pos)
        return len(solutions)

    def is_valid_for_generation(self, board: list[list[int]], row: int, col: int, num: int) -> bool:
        """生成终盘时的普通约束检查"""
        if num in board[row]:
            return False
        for r in range(9):
            if board[r][col] == num:
                return False
        start_row = (row // 3) * 3
        start_col = (col // 3) * 3
        for r in range(start_row, start_row + 3):
            for c in range(start_col, start_col + 3):
                if board[r][c] == num:
                    return False
        return True

    def solve_sudoku_for_generation(self, board: list[list[int]]) -> bool:
        """生成终盘的回溯求解"""
        empty_cells = [(r, c) for r in range(9) for c in range(9) if board[r][c] == 0]
        if not empty_cells:
            return True
        row, col = min(
            empty_cells,
            key=lambda x: len([n for n in range(1, 10) if self.is_valid_for_generation(board, x[0], x[1], n)])
        )
        for num in random.sample(range(1, 10), 9):
            if self.is_valid_for_generation(board, row, col, num):
                board[row][col] = num
                if self.solve_sudoku_for_generation(board):
                    return True
                board[row][col] = 0
        return False

    def generate_normal_sudoku(self) -> list[list[int]] | None:
        """生成普通数独终盘"""
        board = [[0] * 9 for _ in range(9)]
        return board if self.solve_sudoku_for_generation(board) else None

    def generate_puzzle(self) -> tuple[list[list[int]], list[list[int]], int, tuple[int, int]]:
        while True:
            solution = self.generate_normal_sudoku()
            if solution is None:
                continue
            target_box_pos = self.find_max_index_value_subgrid_pos(solution)
            box_row, box_col = target_box_pos
            puzzle = copy.deepcopy(solution)
            all_cells = [(i, j) for i in range(9) for j in range(9)]
            random.shuffle(all_cells)
            max_holes = 64 
            holes = 0
            while holes < 40 and holes < len(all_cells):
                i, j = all_cells[holes]
                original_val = puzzle[i][j]
                if original_val == 0:
                    holes += 1
                    continue  
                puzzle[i][j] = 0
                normal_sol_count = self.check_sudoku_solutions(copy.deepcopy(puzzle), "normal")
                if normal_sol_count == 0:
                    puzzle[i][j] = original_val
                    holes += 1
                    continue
                holes += 1
            current_holes = sum(1 for row in puzzle for cell in row if cell == 0)
            if current_holes < 40:
                continue
            for idx in range(holes, min(max_holes, len(all_cells))):
                i, j = all_cells[idx]
                original_val = puzzle[i][j]
                if original_val == 0:
                    continue  
                puzzle[i][j] = 0
                normal_sol_count = self.check_sudoku_solutions(copy.deepcopy(puzzle), "normal")
                if normal_sol_count == 0:
                    puzzle[i][j] = original_val 
                    continue
                if normal_sol_count == 1:
                    holes += 1 
                    continue
                extremal_sol_count = self.check_sudoku_solutions(
                    copy.deepcopy(puzzle), "extremal", target_box_pos
                )
                if extremal_sol_count == 2:
                    puzzle[i][j] = original_val  
                    continue
                if extremal_sol_count == 1:
                    final_solutions = []
                    self.solve_all_sudoku(
                        copy.deepcopy(puzzle), final_solutions, "extremal", target_box_pos
                    )
                    if len(final_solutions) == 1:
                        holes_count = sum(1 for row in puzzle for cell in row if cell == 0)
                        return solution, puzzle, holes_count, target_box_pos
                puzzle[i][j] = original_val
            final_holes_count = sum(1 for row in puzzle for cell in row if cell == 0)
            if final_holes_count >= 40:
                normal_sol_count = self.check_sudoku_solutions(copy.deepcopy(puzzle), "normal")
                if normal_sol_count >= 2:
                    extremal_sol_count = self.check_sudoku_solutions(
                        copy.deepcopy(puzzle), "extremal", target_box_pos
                    )
                    if extremal_sol_count == 1:
                        final_solutions = []
                        self.solve_all_sudoku(
                            copy.deepcopy(puzzle), final_solutions, "extremal", target_box_pos
                        )
                        if len(final_solutions) == 1:
                            return solution, puzzle, final_holes_count, target_box_pos
 
    def board_to_string(self, board: list[list[int]]) -> str:
        return ' '.join(str(cell) for row in board for cell in row)

    def run(self) -> None:
        """执行生成逻辑：生成指定数量的谜题并写入CSV"""
        for puzzle_id in range(1, self.num_puzzles + 1):
            solution, puzzle, holes_count, target_box_pos = self.generate_puzzle()
            box_row, box_col = target_box_pos
            puzzle_data = {
                "谜题ID": puzzle_id,
                "数独类型": "极值约束数独",
                "终盘数据": self.board_to_string(solution),
                "谜题数据": self.board_to_string(puzzle),
                "挖空数": holes_count,
                "目标最大子网格位置": f"({box_row},{box_col})"
            }
            with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                if os.path.getsize(self.csv_path) == 0:
                    writer.writeheader()
                writer.writerow(puzzle_data)

if __name__ == "__main__":
    NUM_PUZZLES = 5
    CSV_PATH = r"C:\Users\ohhhh\Desktop\sudoku_extremal.csv"
    generator = SudokuExtremalGenerator(num_puzzles=NUM_PUZZLES, csv_path=CSV_PATH)
    generator.run()