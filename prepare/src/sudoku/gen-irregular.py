import random
import copy
import csv
import sys
import time 
from collections import defaultdict
from tqdm import tqdm


class IrregularSudokuGenerator:
    IRREGULAR_BOXES = [
        [0, 1, 2, 3, 9, 10, 18, 27, 28],
        [4, 11, 12, 13, 19, 20, 21, 22, 29],
        [5, 6, 7, 8, 14, 15, 17, 23, 26],
        [16, 24, 25, 33, 34, 35, 42, 43, 44],
        [30, 31, 32, 39, 40, 41, 48, 49, 50],
        [36, 37, 38, 45, 46, 47, 55, 56, 64],
        [51, 58, 59, 60, 61, 67, 68, 69, 76],
        [52, 53, 62, 70, 71, 77, 78, 79, 80],
        [54, 57, 63, 65, 66, 72, 73, 74, 75],
    ]

    def __init__(self):
        self.box_map = [0] * 81
        for box_idx, box in enumerate(self.IRREGULAR_BOXES):
            for pos in box:
                self.box_map[pos] = box_idx
        self.row_col_validator = self._RowColValidator()

    class _RowColValidator:
        def __init__(self):
            self.size = 9 

        def parse_data(self, data_str):
            data = list(map(int, data_str.split(',')))
            return [data[i*self.size : (i+1)*self.size] for i in range(self.size)]

        def is_valid(self, grid, row, col, num):
            for x in range(self.size):
                if grid[row][x] == num:
                    return False
            for x in range(self.size):
                if grid[x][col] == num:
                    return False
            return True

        def find_empty_location(self, grid):
            for i in range(self.size):
                for j in range(self.size):
                    if grid[i][j] == 0:
                        return (i, j)
            return None  

        def solve(self, grid):
            """回溯法求解数独（仅行、列约束），返回所有可能解"""
            empty = self.find_empty_location(grid)
            if not empty:
                return [[row[:] for row in grid]]
            
            row, col = empty
            solutions = []
            for num in range(1, self.size + 1):
                if self.is_valid(grid, row, col, num):
                    grid[row][col] = num
                    for solution in self.solve(grid):
                        solutions.append(solution)
                    grid[row][col] = 0
                    if len(solutions) > 1:
                        return solutions
            return solutions

        def has_multiple_solutions(self, puzzle_data):
            """判断谜题行列约束是否存在多个解"""
            puzzle = self.parse_data(puzzle_data)
            solutions = self.solve(puzzle)
            return len(solutions) > 1
        
    def get_candidates(self, sudoku, pos):
        row = pos // 9  
        col = pos % 9  
        used = set()   

        for c in range(9):
            num = sudoku[row*9 + c]
            if num != 0:
                used.add(num)
        for r in range(9):
            num = sudoku[r*9 + col]
            if num != 0:
                used.add(num)
        box_idx = self.box_map[pos]
        for p in self.IRREGULAR_BOXES[box_idx]:
            num = sudoku[p]
            if num != 0:
                used.add(num)
        return [num for num in range(1, 10) if num not in used]

    def find_next_empty(self, sudoku):
        min_candidates = float('inf')
        target_pos = -1  
        for pos in range(81):
            if sudoku[pos] == 0:
                candidates = self.get_candidates(sudoku, pos)
                if len(candidates) < min_candidates:
                    min_candidates = len(candidates)
                    target_pos = pos
                    if min_candidates == 1:
                        break
        return target_pos

    def _backtrack(self, sudoku):
        pos = self.find_next_empty(sudoku)
        if pos == -1:
            return True 
        candidates = self.get_candidates(sudoku, pos)
        random.shuffle(candidates)

        for num in candidates:
            sudoku[pos] = num
            if self._backtrack(sudoku):
                return True
            sudoku[pos] = 0  
        return False  

    def generate_irregular_sudoku(self):
        sudoku = [0] * 81 
        if self._backtrack(sudoku):
            return sudoku
        else:
            raise RuntimeError("无法生成数独终盘（请检查宫格定义是否正确）")

    def _check_candidate_unique(self, sudoku, sudoku_candidate):
        for i in range(len(sudoku)):
            row = i // 9
            if len(sudoku_candidate[i]) != 1:
                for j in range(9):
                    for value in sudoku_candidate[row * 9 + j]:
                        flag = False
                        for _j in range(9):
                            if _j != j and value in sudoku_candidate[row * 9 + _j]:
                                flag = True
                                break
                        if not flag:
                            sudoku[row * 9 + j] = value
                            sudoku_candidate[row * 9 + j] = [value]
                            break
        for col in range(9):
            for row in range(9):
                i = row * 9 + col
                if sudoku[i] != 0:
                    for j in range(9):
                        for value in sudoku_candidate[j * 9 + col]:
                            flag = False
                            for _j in range(9):
                                if _j != j and value in sudoku_candidate[_j * 9 + col]:
                                    flag = True
                                    break
                            if not flag:
                                sudoku[j * 9 + col] = value  
                                sudoku_candidate[j * 9 + col] = [value]
                                break

        for box in self.IRREGULAR_BOXES:
            value_counts = defaultdict(int)
            for loc in box:
                if sudoku[loc] == 0:
                    for val in sudoku_candidate[loc]:
                        value_counts[val] += 1
            for val, cnt in value_counts.items():
                if cnt == 1:
                    for loc in box:
                        if sudoku[loc] == 0 and val in sudoku_candidate[loc]:
                            sudoku[loc] = val
                            sudoku_candidate[loc] = [val]
                            break

    def _init_sudoku_candidate(self, sudoku, sudoku_candidate):
        for i in range(len(sudoku)):
            if sudoku[i] == 0:
                sudoku_candidate[i] = list(range(1, 10))  
            else:
                sudoku_candidate[i] = [sudoku[i]]

    def _update_sudoku_row(self, sudoku, sudoku_candidate):
        for i in range(len(sudoku)):
            row = i // 9
            row_list = []
            if sudoku[i] != 0:
                for j in range(9):
                    loc = row * 9 + j
                    if sudoku[i] in sudoku_candidate[loc] and len(sudoku_candidate[loc]) > 1:
                        sudoku_candidate[loc].remove(sudoku[i])
                    row_list.append(sudoku[loc])
            if len(sudoku_candidate[i]) == 1:
                sudoku[i] = sudoku_candidate[i][0]
                if row_list.count(sudoku[i]) > 1:
                    raise Exception(f'row {row} error!')  

    def _update_sudoku_col(self, sudoku, sudoku_candidate):
        for col in range(9):
            for row in range(9):
                i = row * 9 + col
                col_list = []
                if sudoku[i] != 0:
                    for j in range(9):
                        loc = j * 9 + col
                        if sudoku[i] in sudoku_candidate[loc] and len(sudoku_candidate[loc]) > 1:
                            sudoku_candidate[loc].remove(sudoku[i])
                        col_list.append(sudoku[loc])
                if len(sudoku_candidate[i]) == 1:
                    sudoku[i] = sudoku_candidate[i][0]
                    if col_list.count(sudoku[i]) > 1:
                        raise Exception(f'col {col} error!')  

    def _update_sudoku_grid(self, sudoku, sudoku_candidate):
        for index in range(81):
            if sudoku[index] == 0:
                continue  
            box_index = self.box_map[index]
            current_box = self.IRREGULAR_BOXES[box_index]
            for loc in current_box:
                if loc == index or sudoku[loc] != 0:
                    continue
                if sudoku[index] in sudoku_candidate[loc] and len(sudoku_candidate[loc]) > 1:
                    sudoku_candidate[loc].remove(sudoku[index])

    def sudoku_solving(self, sudoku, flag=0, solution_cnts=None, sudoku_candidate=None):
        """数独求解与解数量校验：返回解数量状态"""
        if solution_cnts is None:
            solution_cnts = defaultdict(int)  
        if sudoku_candidate is None:
            sudoku_candidate = [[] for _ in range(81)]  
        ret = 0
        self._init_sudoku_candidate(sudoku, sudoku_candidate)
        while True:
            sudoku_candidate_old = copy.deepcopy(sudoku_candidate)
            try:
                self._update_sudoku_row(sudoku, sudoku_candidate)
                self._check_candidate_unique(sudoku, sudoku_candidate)
                self._update_sudoku_col(sudoku, sudoku_candidate)
                self._check_candidate_unique(sudoku, sudoku_candidate)
                self._update_sudoku_grid(sudoku, sudoku_candidate)
                self._check_candidate_unique(sudoku, sudoku_candidate)
            except Exception as e:  
                ret = -2 
                break
            if 0 not in sudoku:
                solution_cnts[flag] += 1
                ret = solution_cnts[flag]
                break
            if sudoku_candidate == sudoku_candidate_old:
                ret = -1
                break
        if ret == -1:
            first0 = sudoku.index(0)
            candidates = sudoku_candidate[first0]
            for v in candidates:
                sudoku2 = sudoku.copy()
                sudoku2[first0] = v
                new_sudoku_candidate = copy.deepcopy(sudoku_candidate)
                new_sudoku_candidate[first0] = [v]
                ret = self.sudoku_solving(sudoku2, flag, solution_cnts, new_sudoku_candidate)
                if solution_cnts[flag] > 2:
                    break
        if solution_cnts[flag] == 1:
            ret = 0
        return ret

    def sudoku_puzzle_dibble(self, sudoku):
        """挖空生成谜题：在终盘基础上挖空，确保挖空后仍有唯一解"""
        sudoku_dibble = sudoku.copy()
        enable_choice = list(range(81))  
        i = 0  
        j = 0  
        flag = 0 
        random.shuffle(enable_choice)  
        target = random.randint(35, 40)  
        while enable_choice and i < target:
            random_index = enable_choice[j]
            if sudoku_dibble[random_index] == 0:
                j = (j + 1) % len(enable_choice)
                continue
            original_value = sudoku_dibble[random_index]
            sudoku_dibble[random_index] = 0
            sudoku_tmp = sudoku_dibble.copy()
            ret = self.sudoku_solving(sudoku_tmp, flag)
            flag += 1
            if ret == 0:  
                i += 1
                enable_choice.remove(random_index)  
                random.shuffle(enable_choice) 
                j = 0  
                if i >= target:
                    break
            else:  
                sudoku_dibble[random_index] = original_value
                j = (j + 1) % len(enable_choice)
                if j == 0:
                    break
        return sudoku_dibble

    def generate_and_save(self, output_path, num_puzzles=4):
        """批量生成符合条件的锯齿数独"""
        saved_puzzles = 0  
        attempts = 0       
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['谜题ID', '数独类型', '终盘数据', '谜题数据'])
            pbar = tqdm(total=num_puzzles, desc="生成符合条件的锯齿数独")

            while saved_puzzles < num_puzzles:
                attempts += 1
                start_time = time.time() 
                try:
                    sudoku_final = self.generate_irregular_sudoku()
                    sudoku_puzzle = self.sudoku_puzzle_dibble(sudoku_final)
                    final_str = ','.join(map(str, sudoku_final))
                    puzzle_str = ','.join(map(str, sudoku_puzzle))
                    has_multiple = self.row_col_validator.has_multiple_solutions(puzzle_str)
                    elapsed_time = time.time() - start_time
                    if elapsed_time > 900:
                        tqdm.write(f"生成超时 {elapsed_time:.2f}s，跳过（尝试{attempts}）")
                        continue
                    if has_multiple:
                        saved_puzzles += 1
                        writer.writerow([
                            saved_puzzles,
                            "锯齿数独",
                            final_str,
                            puzzle_str
                        ])
                        pbar.update(1)
                        tqdm.write(f"保存第{saved_puzzles}个谜题（尝试{attempts}，耗时{elapsed_time:.2f}s）")

                except Exception as e:
                    tqdm.write(f"生成出错：{str(e)}，跳过（尝试{attempts}）")
                    continue
            pbar.close()


def main():
    if len(sys.argv) > 1:
        output_path = sys.argv[1]
    else:
        output_path = input("请输入输出CSV文件路径：")
    generator = IrregularSudokuGenerator()
    generator.generate_and_save(output_path, num_puzzles=100)


if __name__ == '__main__':
    main()