import copy
import random
import csv

class DiscontinuousSudokuGenerator:
    def find_empty(self, board):
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] == 0:
                    return (i, j)  
        return None
    def is_valid_normal(self, board, num, pos):
        row, col = pos
        for i in range(9):
            if board[row][i] == num and col != i:
                return False
        for i in range(9):
            if board[i][col] == num and row != i:
                return False
        box_x = col // 3
        box_y = row // 3
        for i in range(box_y*3, box_y*3 + 3):
            for j in range(box_x*3, box_x*3 + 3):
                if board[i][j] == num and (i,j) != pos:
                    return False
        return True

    def is_valid_discontinuous(self, board, num, pos):
        row, col = pos
        if not self.is_valid_normal(board, num, pos):
            return False
        if col > 0 and board[row][col - 1] != 0:
            if abs(board[row][col - 1] - num) == 1:
                return False
        if col < 8 and board[row][col + 1] != 0:
            if abs(board[row][col + 1] - num) == 1:
                return False
        if row > 0 and board[row - 1][col] != 0:
            if abs(board[row - 1][col] - num) == 1:
                return False
        if row < 8 and board[row + 1][col] != 0:
            if abs(board[row + 1][col] - num) == 1:
                return False
        return True

    def solve_all_sudoku(self, board, solutions, constraint_type="normal"):
        find = self.find_empty(board)
        if not find:
            solutions.append(copy.deepcopy(board))
            return
        else:
            row, col = find
        for num in range(1, 10):
            if constraint_type == "normal":
                valid = self.is_valid_normal(board, num, (row, col))
            else:  
                valid = self.is_valid_discontinuous(board, num, (row, col))
                
            if valid:
                board[row][col] = num
                self.solve_all_sudoku(board, solutions, constraint_type)
                board[row][col] = 0  

    def limit_solve(self, board, solutions, max_solutions=2, constraint_type="normal"):
        if len(solutions) >= max_solutions:
            return
        
        find = self.find_empty(board)
        if not find:
            solutions.append(copy.deepcopy(board))
            return
        else:
            row, col = find

        for num in range(1, 10):
            if constraint_type == "normal":
                valid = self.is_valid_normal(board, num, (row, col))
            else: 
                valid = self.is_valid_discontinuous(board, num, (row, col))
                
            if valid:
                board[row][col] = num
                self.limit_solve(board, solutions, max_solutions, constraint_type)
                board[row][col] = 0
                if len(solutions) >= max_solutions:
                    return

    def check_sudoku_solutions(self, board, constraint_type="normal"):
        board_copy = copy.deepcopy(board)
        solutions = []
        self.limit_solve(board_copy, solutions, 2, constraint_type)
        if len(solutions) == 0:
            return 0
        elif len(solutions) == 1:
            return 1
        else:
            return 2

    def solve_sudoku_for_generation(self, board):
        for row in range(9):
            for col in range(9):
                if board[row][col] == 0:
                    for num in random.sample(range(1, 10), 9):
                        if self.is_valid_discontinuous(board, num, (row, col)):
                            board[row][col] = num
                            if self.solve_sudoku_for_generation(board):
                                return True
                            board[row][col] = 0  
                    return False 
        return True  

    def generate_discontinuous_sudoku(self):
        """生成满足不连续性约束的数独终盘"""
        for _ in range(10):
            board = [[0 for _ in range(9)] for _ in range(9)]
            if self.solve_sudoku_for_generation(board):
                return board
        raise RuntimeError("无法生成有效终盘，请检查约束条件")

    def generate_puzzle(self):
        while True:
            solution = self.generate_discontinuous_sudoku()
            if solution is None:
                continue
            puzzle = copy.deepcopy(solution)
            cells = [(i, j) for i in range(9) for j in range(9)]
            random.shuffle(cells)
            min_holes = 40 
            max_holes = 46  
            holes = 0
            for (i, j) in cells:
                if holes >= max_holes:
                    break 
                if puzzle[i][j] == 0:
                    continue
                original_value = puzzle[i][j]
                puzzle[i][j] = 0
                normal_count = self.check_sudoku_solutions(copy.deepcopy(puzzle), "normal")
                if normal_count == 0:
                    puzzle[i][j] = original_value
                    continue
                if normal_count == 1:
                    holes += 1
                    continue
                if normal_count == 2:
                    discontinuous_count = self.check_sudoku_solutions(copy.deepcopy(puzzle), "discontinuous")
                    if discontinuous_count == 2:
                        puzzle[i][j] = original_value
                        continue
                    if discontinuous_count == 1:
                        solutions = []
                        self.solve_all_sudoku(copy.deepcopy(puzzle), solutions, "discontinuous")
                        if len(solutions) == 1:
                            holes += 1
            
            holes_count = sum(1 for row in puzzle for cell in row if cell == 0)
            if holes_count >= min_holes:
                return solution, puzzle, holes_count
            
    def board_to_string(self, board):
        return ' '.join(str(cell) for row in board for cell in row)


    def main(self, num_puzzles=5, output_path=r"sudoku_constraint.csv"):
        puzzles = []
        for i in range(num_puzzles):
            print(f"生成第 {i+1} 个谜题...")
            solution, puzzle, holes_count = self.generate_puzzle()
            puzzles.append({
                "id": i + 1,
                "type": "不连续性约束数独",
                "solution": self.board_to_string(solution),
                "puzzle": self.board_to_string(puzzle),
                "holes": holes_count
            })
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
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
    generator = DiscontinuousSudokuGenerator()
    generator.main()