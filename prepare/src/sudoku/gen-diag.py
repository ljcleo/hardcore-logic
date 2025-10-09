import copy
import random
import csv

class DiagSudokuGenerator:
    @staticmethod
    def find_empty(board):
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] == 0:
                    return (i, j)  
        return None
    @staticmethod
    def is_valid_normal(board, num, pos):
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
                if board[i][j] == num and (i,j) != pos:
                    return False
        return True

    @staticmethod
    def is_valid_diagonal(board, num, pos):
        """检查否符合对角线约束"""
        if not DiagSudokuGenerator.is_valid_normal(board, num, pos):
            return False
        row, col = pos
        if row == col:
            for i in range(9):
                if board[i][i] == num and (i, i) != pos:
                    return False
        if row + col == 8:
            for i in range(9):
                if board[i][8-i] == num and (i, 8-i) != pos:
                    return False
        return True

    @staticmethod
    def solve_all_sudoku(board, solutions, constraint_type="normal"):
        """求解数独所有解，存储到solutions列表"""
        find = DiagSudokuGenerator.find_empty(board)
        if not find:
            solutions.append(copy.deepcopy(board))
            return
        else:
            row, col = find
        for num in range(1, 10):
            if constraint_type == "normal":
                valid = DiagSudokuGenerator.is_valid_normal(board, num, (row, col))
            else:  
                valid = DiagSudokuGenerator.is_valid_diagonal(board, num, (row, col))          
            if valid:
                board[row][col] = num
                DiagSudokuGenerator.solve_all_sudoku(board, solutions, constraint_type)
                board[row][col] = 0

    @staticmethod
    def limit_solve(board, solutions, max_solutions=2, constraint_type="normal"):
        """求解数独，最多找到max_solutions个解后停止"""
        if len(solutions) >= max_solutions:
            return
        find = DiagSudokuGenerator.find_empty(board)
        if not find:
            solutions.append(copy.deepcopy(board))
            return
        else:
            row, col = find
        for num in range(1, 10):
            if constraint_type == "normal":
                valid = DiagSudokuGenerator.is_valid_normal(board, num, (row, col))
            else: 
                valid = DiagSudokuGenerator.is_valid_diagonal(board, num, (row, col))
            if valid:
                board[row][col] = num
                DiagSudokuGenerator.limit_solve(board, solutions, max_solutions, constraint_type)
                board[row][col] = 0
                if len(solutions) >= max_solutions:
                    return

    @staticmethod
    def check_sudoku_solutions(board, constraint_type="normal"):
        board_copy = copy.deepcopy(board)
        solutions = []
        DiagSudokuGenerator.limit_solve(board_copy, solutions, 2, constraint_type)
        if len(solutions) == 0:
            return 0
        elif len(solutions) == 1:
            return 1
        else:  
            return 2

    @staticmethod
    def is_valid_for_generation(board, row, col, num):
        """生成终盘时的有效性验证（含对角线约束）"""
        if num in board[row]:
            return False
        for r in range(9):
            if board[r][col] == num:
                return False
        start_row = (row // 3) * 3
        start_col = (col // 3) * 3
        for r in range(start_row, start_row+3):
            for c in range(start_col, start_col+3):
                if board[r][c] == num:
                    return False
        if row == col:
            for i in range(9):
                if board[i][i] == num:
                    return False
        if row + col == 8:
            for i in range(9):
                if board[i][8 - i] == num:
                    return False
        return True

    @staticmethod
    def solve_sudoku_for_generation(board):
        """生成终盘专用求解函数"""
        empty_cells = [(r, c) for r in range(9) for c in range(9) if board[r][c] == 0]
        if not empty_cells:
            return True 
        row, col = min(empty_cells, key=lambda x: len([n for n in range(1,10) 
                                                      if DiagSudokuGenerator.is_valid_for_generation(board, x[0], x[1], n)]))
        for num in random.sample(range(1, 10), 9):
            if DiagSudokuGenerator.is_valid_for_generation(board, row, col, num):
                board[row][col] = num
                if DiagSudokuGenerator.solve_sudoku_for_generation(board):
                    return True
                board[row][col] = 0
        return False  

    @staticmethod
    def generate_diagonal_sudoku():
        """生成对角线约束的数独终盘"""
        board = [[0 for _ in range(9)] for _ in range(9)]
        if DiagSudokuGenerator.solve_sudoku_for_generation(board):
            return board
        else:
            return None

    @staticmethod
    def generate_puzzle():
        while True:
            solution = DiagSudokuGenerator.generate_diagonal_sudoku()
            if solution is None:
                continue  
            puzzle = copy.deepcopy(solution)
            cells = [(i, j) for i in range(9) for j in range(9)]
            random.shuffle(cells)
            holes = 0
            max_holes = 64
            for (i, j) in cells:
                if holes >= max_holes:
                    break
                original_value = puzzle[i][j]
                if original_value == 0:
                    continue
                puzzle[i][j] = 0
                solutions_count_normal = DiagSudokuGenerator.check_sudoku_solutions(
                    copy.deepcopy(puzzle), "normal"
                )
                if solutions_count_normal == 0:
                    puzzle[i][j] = original_value
                    continue
                if solutions_count_normal == 1:
                    holes += 1
                    continue
                solutions_count_diagonal = DiagSudokuGenerator.check_sudoku_solutions(
                    copy.deepcopy(puzzle), "diagonal"
                )
                if solutions_count_diagonal == 2:
                    puzzle[i][j] = original_value
                    continue
                if solutions_count_diagonal == 1:
                    solutions = []
                    DiagSudokuGenerator.solve_all_sudoku(copy.deepcopy(puzzle), solutions, "diagonal")
                    if len(solutions) == 1:
                        current_holes = sum(1 for row in puzzle for cell in row if cell == 0)
                        if current_holes >= 40:
                            return solution, puzzle, current_holes
                        else:
                            puzzle[i][j] = original_value
                            continue

    @staticmethod
    def board_to_string(board):
        return ' '.join(str(cell) for row in board for cell in row)

    @classmethod
    def main(cls, num_puzzles=50, save_path=r"9x9sudoku_extremal.csv"):
        puzzles = []
        for i in range(num_puzzles):
            solution, puzzle, holes_count = cls.generate_puzzle()
            puzzles.append({
                "id": i + 1,
                "type": "对角线约束数独",
                "solution": cls.board_to_string(solution),
                "puzzle": cls.board_to_string(puzzle),
                "holes": holes_count
            })
        with open(save_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['谜题ID', '数独类型', '终盘数据', '谜题数据', '挖空数']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for p in puzzles:
                writer.writerow({
                    '谜题ID': p['id'],
                    '数独类型': p['type'],
                    '终盘数据': p['solution'],
                    '谜题数据': p['puzzle'],
                    '挖空数': p['holes']
                })


if __name__ == "__main__":
    DiagSudokuGenerator.main(num_puzzles=50)