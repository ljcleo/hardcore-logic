import random
from collections import defaultdict
import csv

class SudokuGenerator:
    def __init__(self, size=9, grid_size=3, max_holes=None, csv_path=None, puzzle_count=50):
        self.size = size
        self.grid_size = grid_size
        self.total_cells = size * size
        self.max_holes = max_holes or (size == 9 and 81 or 116)  
        #最大挖空数，默认为挖到不能挖为止。size=16时，设置为115较好
        self.csv_path = csv_path
        self.puzzle_count = puzzle_count

    def check_candidate_unique(self, sudoku, sudoku_candidate):
        N = self.size
        G = self.grid_size
        for i in range(len(sudoku)):
            row = i // N
            if len(sudoku_candidate[i]) != 1:
                for j in range(N):
                    for value in sudoku_candidate[row * N + j]:
                        flag = False
                        for _j in range(N):
                            if _j != j and value in sudoku_candidate[row * N + _j]:
                                flag = True
                                break
                        if not flag:
                            sudoku[row * N + j] = value
                            sudoku_candidate[row * N + j] = [value]
                            break
        for col in range(N):
            for row in range(N):
                i = row * N + col
                if sudoku[i] != 0:
                    for j in range(N):
                        for value in sudoku_candidate[j * N + col]:
                            flag = False
                            for _j in range(N):
                                if _j != j and value in sudoku_candidate[_j * N + col]:
                                    flag = True
                                    break
                            if not flag:
                                sudoku[j * N + col] = value
                                sudoku_candidate[j * N + col] = [value]
                                break
        for i in range(len(sudoku)):
            row = i // N
            col = i % N
            if sudoku[i] != 0:
                for k in range(G):
                    for t in range(G):
                        x = (row // G * G + k) * N
                        y = (col // G * G + t)
                        loc = x + y
                        for value in sudoku_candidate[loc]:
                            flag = False
                            for _k in range(G):
                                for _t in range(G):
                                    if (_k != k or _t != t) and value in sudoku_candidate[(row//G*G + _k)*N + (col//G*G + _t)]:
                                        flag = True
                                        break
                                if flag:
                                    break
                            if not flag:
                                sudoku[loc] = value
                                sudoku_candidate[loc] = [value]
                                break

    def init_sudoku_candidate(self, sudoku, sudoku_candidate):
        N = self.size
        for i in range(len(sudoku)):
            sudoku_candidate[i] = [sudoku[i]] if sudoku[i] != 0 else list(range(1, N + 1))

    def update_sudoku_row(self, sudoku, sudoku_candidate):
        N = self.size
        for i in range(len(sudoku)):
            row = i // N
            row_list = []
            if sudoku[i] != 0:
                for j in range(N):
                    loc = row * N + j
                    if sudoku[i] in sudoku_candidate[loc] and len(sudoku_candidate[loc]) > 1:
                        sudoku_candidate[loc].remove(sudoku[i])
                    row_list.append(sudoku[loc])
            if len(sudoku_candidate[i]) == 1:
                sudoku[i] = sudoku_candidate[i][0]
                if row_list.count(sudoku[i]) > 1:
                    raise Exception(f'row {row} error!')

    def update_sudoku_col(self, sudoku, sudoku_candidate):
        N = self.size
        for col in range(N):
            for row in range(N):
                i = row * N + col
                col_list = []
                if sudoku[i] != 0:
                    for j in range(N):
                        loc = j * N + col
                        if sudoku[i] in sudoku_candidate[loc] and len(sudoku_candidate[loc]) > 1:
                            sudoku_candidate[loc].remove(sudoku[i])
                        col_list.append(sudoku[loc])
                if len(sudoku_candidate[i]) == 1:
                    sudoku[i] = sudoku_candidate[i][0]
                    if col_list.count(sudoku[i]) > 1:
                        raise Exception(f'col {col} error!')

    def update_sudoku_grid(self, sudoku, sudoku_candidate):
        N = self.size
        G = self.grid_size
        for i in range(len(sudoku)):
            row = i // N
            col = i % N
            grid_list = []
            if sudoku[i] != 0:
                for k in range(G):
                    for t in range(G):
                        x = (row // G * G + k) * N
                        y = (col // G * G + t)
                        loc = x + y
                        if sudoku[i] in sudoku_candidate[loc] and len(sudoku_candidate[loc]) > 1:
                            sudoku_candidate[loc].remove(sudoku[i])
                        grid_list.append(sudoku[loc])
            if len(sudoku_candidate[i]) == 1:
                sudoku[i] = sudoku_candidate[i][0]
                if grid_list.count(sudoku[i]) > 1:
                    raise Exception(f'grid error!')

    def sudoku_solving(self, sudoku, flag=0, solution_cnts=None, sudoku_candidate=None):
        N = self.size
        if solution_cnts is None:
            solution_cnts = defaultdict(int)
        if sudoku_candidate is None:
            sudoku_candidate = [[] for _ in range(self.total_cells)]
        ret = 0
        self.init_sudoku_candidate(sudoku, sudoku_candidate)
        while True:
            sudoku_candidate_old = [c.copy() for c in sudoku_candidate]
            try:
                self.update_sudoku_row(sudoku, sudoku_candidate)
                self.check_candidate_unique(sudoku, sudoku_candidate)
                self.update_sudoku_col(sudoku, sudoku_candidate)
                self.check_candidate_unique(sudoku, sudoku_candidate)
                self.update_sudoku_grid(sudoku, sudoku_candidate)
                self.check_candidate_unique(sudoku, sudoku_candidate)
            except Exception:
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
            for v in sudoku_candidate[first0]:
                sudoku2 = sudoku.copy()
                sudoku2[first0] = v
                new_cand = [c.copy() for c in sudoku_candidate]
                new_cand[first0] = [v]
                ret = self.sudoku_solving(sudoku2, flag, solution_cnts, new_cand)
                if solution_cnts[flag] > 2:
                    break
        if solution_cnts[flag] == 1:
            ret = 0
        return ret

    def get_random_num(self, sudoku, index):
        N = self.size
        G = self.grid_size
        candidate = list(range(1, N + 1))
        row = index // N
        col = index % N
        grid_x = (row // G) * G
        grid_y = (col // G) * G
        for j in range(col):
            if sudoku[row * N + j] in candidate:
                candidate.remove(sudoku[row * N + j])
        for i in range(row):
            if sudoku[i * N + col] in candidate:
                candidate.remove(sudoku[i * N + col])
        for _i in range(G):
            for _j in range(G):
                if sudoku[(grid_x + _i) * N + (grid_y + _j)] in candidate:
                    candidate.remove(sudoku[(grid_x + _i) * N + (grid_y + _j)])
        return random.choice(candidate) if candidate else -1

    def sudoku_generate_backtracking(self):
        N = self.size
        G = self.grid_size
        if N == 9:
            while True:
                sudoku = [0] * 81
                i = 0
                while i < 81:
                    num = self.get_random_num(sudoku, i)
                    if num == -1:
                        break
                    sudoku[i] = num
                    i += 1
                if i == 81:
                    return sudoku
        else:
            row_mask = [0] * N
            col_mask = [0] * N
            grid_mask = [0] * N
            sudoku = [0] * (N * N)
            grid_index = [((i//N)//G * G + (i%N)//G) for i in range(N * N)]

            def get_candidates(pos):
                row, col, grid = pos // N, pos % N, grid_index[pos]
                used = row_mask[row] | col_mask[col] | grid_mask[grid]
                return [num for num in range(1, N + 1) if not (used & (1 << (num - 1)))]

            def backtrack(positions):
                if not positions:
                    return True
                pos = min(positions, key=lambda p: len(get_candidates(p)))
                candidates = get_candidates(pos)
                random.shuffle(candidates)
                for num in candidates:
                    row, col, grid = pos // N, pos % N, grid_index[pos]
                    sudoku[pos] = num
                    mask = 1 << (num - 1)
                    row_mask[row] |= mask
                    col_mask[col] |= mask
                    grid_mask[grid] |= mask
                    new_positions = [p for p in positions if p != pos]
                    if backtrack(new_positions):
                        return True
                    sudoku[pos] = 0
                    row_mask[row] ^= mask
                    col_mask[col] ^= mask
                    grid_mask[grid] ^= mask
                return False

            positions = list(range(N * N))
            random.shuffle(positions)
            if backtrack(positions):
                return sudoku
            else:
                return self.sudoku_generate_backtracking()

    def sudoku_puzzle_dibble(self, sudoku):
        sudoku_dibble = sudoku.copy()
        enable_choice = list(range(self.total_cells))
        i = 0
        j = 0
        flag = 0
        random.shuffle(enable_choice)
        while enable_choice and i < self.max_holes:
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
                if i >= self.max_holes:
                    break
            else:
                sudoku_dibble[random_index] = original_value
                j = (j + 1) % len(enable_choice)
                if j == 0:
                    break
        return sudoku_dibble

    def out_sudoku_csv(self, sudoku):
        N = self.size
        for i in range(len(sudoku)):
            print(f'{sudoku[i]},', end='')
            if (i + 1) % N == 0:
                print()
        print('-' * 50)

    def format_sudoku(self, sudoku):
        N = self.size
        G = self.grid_size
        lines = []
        for row in range(N):
            line_parts = []
            for col in range(N):
                num = sudoku[row * N + col]
                line_parts.append(f"{num:2}")
                if (col + 1) % G == 0 and col != N - 1:
                    line_parts.append("|")
            line_str = " ".join(line_parts).strip()
            lines.append(line_str)
            if (row + 1) % G == 0 and row != N - 1:
                sep = "+".join(["-" * (3 * G - 1)] * G)
                lines.append(sep)
        return "\n".join(lines)

    def generate_to_csv(self):
        with open(self.csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['谜题ID', '终盘数据', '谜题数据']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for puzzle_id in range(1, self.puzzle_count + 1):
                sudoku_final = self.sudoku_generate_backtracking()
                sudoku_puzzle = self.sudoku_puzzle_dibble(sudoku_final)
                writer.writerow({
                    '谜题ID': puzzle_id,
                    '终盘数据': ','.join(map(str, sudoku_final)),
                    '谜题数据': ','.join(map(str, sudoku_puzzle))
                })

if __name__ == "__main__":
    SudokuGenerator(
        size=9, 
        grid_size=3, 
        csv_path=r"sudoku.csv", 
        puzzle_count=50
    ).generate_to_csv()