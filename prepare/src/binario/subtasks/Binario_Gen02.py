import random
from z3 import *
from itertools import combinations

'''
- task: Binario
- subtask: Give more conflicting constraints
- introduction: Convert some digits into constraints for presentation

'''
class Binario_Gen02:
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

    def _has_unique_solution(self, puzzle):
        solution_count = 0

        def backtrack(row, col):
            nonlocal solution_count
            if solution_count >= 2:  # 如果已经找到两个解，提前终止
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
        return solution_count == 1

    def _is_valid_move_puzzle(self, puzzle, row, col, num):
        # 检查行和列中0和1的数量不超过一半
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

    def create_hard_puzzle(self):
        puzzle, solution = self.create_puzzle()
        removed_cells = []

        filled_cells = [(i, j) for i in range(self.size)
                        for j in range(self.size)
                        if puzzle[i][j] is not None]

        num_removals = max(1, len(filled_cells) // 2)
        to_remove = random.sample(filled_cells, num_removals)

        for row, col in to_remove:
            removed_cells.append({
                'position': (row, col),
                'value': puzzle[row][col]
            })
            puzzle[row][col] = None

        constraints = self._generate_constraints(removed_cells)

        minimal_constraints = self._find_minimal_constraints(removed_cells, constraints)

        return puzzle, solution, removed_cells, minimal_constraints

    def _generate_constraints(self, removed_cells):
        constraints = []
        pos_to_val = {cell['position']: cell['value'] for cell in removed_cells}

        # 1. 选择性生成值约束（最多30%单元格）
        for pos, val in random.sample(list(pos_to_val.items()),
                                      k=max(1, len(pos_to_val) // 3)):
            constraints.append({
                'type': 'value',
                'target': pos,
                'value': val,
                'description': f"X[{pos[0]},{pos[1]}] == {val}"
            })

        for (pos1, val1), (pos2, val2) in combinations(pos_to_val.items(), 2):
            if val1 == val2:
                constraints.append({
                    'type': 'equal',
                    'target1': pos1,
                    'target2': pos2,
                    'description': f"X[{pos1[0]},{pos1[1]}] == X[{pos2[0]},{pos2[1]}]"
                })
            else:
                constraints.append({
                    'type': 'not_equal',
                    'target1': pos1,
                    'target2': pos2,
                    'description': f"X[{pos1[0]},{pos1[1]}] != X[{pos2[0]},{pos2[1]}]"
                })

        if len(pos_to_val) >= 4:
            for cells in combinations(pos_to_val.items(), 4):
                (a_pos, a_val), (b_pos, b_val), (c_pos, c_val), (d_pos, d_val) = cells
                if a_val + b_val == c_val + d_val:
                    constraints.append({
                        'type': 'sum_equal',
                        'targets': [a_pos, b_pos, c_pos, d_pos],
                        'description': f"X[{a_pos[0]},{a_pos[1]}] + X[{b_pos[0]},{b_pos[1]}] == "
                                       f"X[{c_pos[0]},{c_pos[1]}] + X[{d_pos[0]},{d_pos[1]}]"
                    })

        if not self._verify_self_consistent(constraints, pos_to_val):
            print("Warning: Generated constraints are inconsistent, regenerating...")
            return self._generate_constraints(removed_cells)

        return constraints

    def _verify_self_consistent(self, constraints, pos_to_val):
        for c in constraints:
            if c['type'] == 'value':
                if pos_to_val[c['target']] != c['value']:
                    return False
            elif c['type'] == 'equal':
                if pos_to_val[c['target1']] != pos_to_val[c['target2']]:
                    return False
        return True

    def _find_minimal_constraints(self, removed_cells, constraints):

        if not self._verify_constraints_with_z3(removed_cells, constraints):
            value_constraints = [c for c in constraints if c['type'] == 'value']
            if not self._verify_constraints_with_z3(removed_cells, value_constraints):
                raise ValueError("Value constraints are also insufficient")
            print("Falling back to pure value constraint sets")
            return value_constraints

        # 按优先级排序
        CONSTRAINT_PRIORITY = {
            'sum_equal': 1, 'sum_greater': 1, 'sum_less': 1,
            'equal': 1, 'not_equal': 1, 'greater': 1, 'less': 1,
            'value': 1
        }

        sorted_constraints = sorted(
            constraints,
            key=lambda x: (CONSTRAINT_PRIORITY[x['type']], random.random())
        )

        minimal_set = list(sorted_constraints)
        changed = True
        iteration = 0

        while changed:
            iteration += 1
            changed = False

            for constraint in list(minimal_set):
                temp_set = [c for c in minimal_set if c != constraint]

                if self._verify_constraints_with_z3(removed_cells, temp_set):
                    minimal_set = temp_set
                    changed = True
                    break

        return minimal_set

    def _verify_constraints_with_z3(self, removed_cells, constraints):
        solver = Solver()
        var_dict = {cell['position']: Int(f'x_{cell["position"][0]}_{cell["position"][1]}')
                    for cell in removed_cells}

        for var in var_dict.values():
            solver.add(Or(var == 0, var == 1))

        constraint_groups = {
            'sum': [c for c in constraints if c['type'].startswith('sum_')],
            'relation': [c for c in constraints if c['type'] in ('equal', 'not_equal', 'greater', 'less')],
            'value': [c for c in constraints if c['type'] == 'value']
        }

        for constraint in constraint_groups['sum']:
            pos_a, pos_b, pos_c, pos_d = constraint['targets']
            sum1 = var_dict[pos_a] + var_dict[pos_b]
            sum2 = var_dict[pos_c] + var_dict[pos_d]
            if constraint['type'] == 'sum_equal':
                solver.add(sum1 == sum2)
            elif constraint['type'] == 'sum_greater':
                solver.add(sum1 > sum2)
            elif constraint['type'] == 'sum_less':
                solver.add(sum1 < sum2)

        for constraint in constraint_groups['relation']:
            pos1, pos2 = constraint['target1'], constraint['target2']
            if constraint['type'] == 'equal':
                solver.add(var_dict[pos1] == var_dict[pos2])
            elif constraint['type'] == 'not_equal':
                solver.add(var_dict[pos1] != var_dict[pos2])
            elif constraint['type'] == 'greater':
                solver.add(var_dict[pos1] > var_dict[pos2])
            elif constraint['type'] == 'less':
                solver.add(var_dict[pos1] < var_dict[pos2])

        for constraint in constraint_groups['value']:
            pos = constraint['target']
            solver.add(var_dict[pos] == constraint['value'])

        solutions = []
        while solver.check() == sat and len(solutions) < 2:
            model = solver.model()
            solution = {pos: model[var].as_long() for pos, var in var_dict.items()}
            solutions.append(solution)
            solver.add(Or([var != solution[pos] for pos, var in var_dict.items()]))

        if len(solutions) != 1:
            return False
        original = {cell['position']: cell['value'] for cell in removed_cells}
        return solutions[0] == original

    def print_board(self, board, constraints=None, removed_cells=None):
        for i in range(self.size):
            line = []
            for j in range(self.size):
                if board[i][j] is not None:
                    line.append(str(board[i][j]))
                else:
                    line.append('*' if removed_cells and (i, j) in [c['position'] for c in removed_cells] else '_')


    def _format_puzzle(self, puzzle):
        lines = []
        for row in puzzle:
            line = []
            for cell in row:
                line.append('.' if cell is None else str(cell))
            lines.append(' '.join(line))
        return '\n'.join(lines)

    def _format_constraints(self, constraints):
        lines = []
        for c in constraints:
            if c['type'] == 'value':
                r, c_ = c['target']
                lines.append(f"- ({r+1}, {c_+1}) == {c['value']}")
            elif c['type'] == 'equal':
                r1, c1 = c['target1']
                r2, c2 = c['target2']
                lines.append(f"- ({r1+1}, {c1+1}) == ({r2+1}, {c2+1})")
            elif c['type'] == 'not_equal':
                r1, c1 = c['target1']
                r2, c2 = c['target2']
                lines.append(f"- ({r1+1}, {c1+1}) != ({r2+1}, {c2+1})")
            elif c['type'] == 'sum_equal':
                (a_r,a_c),(b_r,b_c),(c_r,c_c),(d_r,d_c) = c['targets']
                lines.append(f"- ({a_r+1}, {a_c+1}) + ({b_r+1}, {b_c+1}) == "
                             f"({c_r+1}, {c_c+1}) + ({d_r+1}, {d_c+1})")
            else:
                lines.append("None")
        return "\n".join(lines)

    def generate_formatted_puzzle(self, size=8):
        self.size = size
        self.half = size // 2

        puzzle, solution, removed_cells, minimal_constraints = self.create_hard_puzzle()

        # 拼接 puzzle 和 constraints
        puzzle_text = self._format_puzzle(puzzle)
        constraints_text = self._format_constraints(minimal_constraints)

        combined_text = puzzle_text
        if constraints_text:  # 如果有额外约束
            combined_text += "\n\n### Extra Clues (Index Start from 1):\n" + constraints_text

        formatted = {
            'puzzle': combined_text,
            'solution': solution,
        }
        return formatted


# 使用示例
if __name__ == "__main__":
    generator = Binario_Gen02(size=6)

    # 方法1：获取格式化后的所有内容
    puzzle_data = generator.generate_formatted_puzzle(size=6)
    print("获取到的谜题数据:")
    print(puzzle_data['puzzle'])
    print(puzzle_data['solution'])


