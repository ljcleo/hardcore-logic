import random
import time
import multiprocessing
from multiprocessing import Queue
from collections import deque
import pandas as pd
from queue import Empty  
import pyarrow.parquet as pq
import pyarrow as pa


class MinesweeperGenerator:
    def __init__(self, rows=12, cols=12, unknown_range=(0.75,0.85),
                 difficulty="hard", timeout=180, min_mines=4, max_mines=12,
                 use_letters=False):  
        self.rows = rows
        self.cols = cols
        self.unknown_range = unknown_range
        self.difficulty = difficulty
        self.timeout = timeout
        self.min_mines = min_mines
        self.max_mines = max_mines
        self.use_letters = use_letters  
        self.id_counter = 1
        self.data = []

    def format_puzzle_as_letters(self, puzzle):
        mapping = {0: "Z", 1:"A", 2:"B", 3:"C", 4:"D", 5:"E", 6:"F", 7:"G", 8:"H", -2:"."}
        formatted_rows = []
        for row in puzzle:
            formatted_row = [mapping.get(cell, str(cell)) for cell in row]
            formatted_rows.append(" ".join(formatted_row))
        return "\n".join(formatted_rows)

    def get_neighbors(self, r, c):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    yield (nr, nc)

    def find_definite_mines(self, grid, max_solutions=None, verbose=False):
        rows, cols = self.rows, self.cols
        known_flags = set()
        unknown_cells = []
        coord_to_var = {}

        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == -1:
                    known_flags.add((r, c))
                elif grid[r][c] == -2:
                    idx = len(unknown_cells)
                    unknown_cells.append((r, c))
                    coord_to_var[(r, c)] = idx

        constraints = []
        for r in range(rows):
            for c in range(cols):
                val = grid[r][c]
                if isinstance(val, int) and 0 <= val <= 8:
                    adj_unknowns = []
                    adj_known_flags = 0
                    for nr, nc in self.get_neighbors(r, c):
                        if (nr, nc) in coord_to_var:
                            adj_unknowns.append(coord_to_var[(nr, nc)])
                        elif (nr, nc) in known_flags:
                            adj_known_flags += 1
                    if adj_unknowns:
                        required = val - adj_known_flags
                        constraints.append({'cell': (r, c), 'vars': adj_unknowns, 'required': required})

        if not constraints:
            return set()

        var_adj = [set() for _ in range(len(unknown_cells))]
        for cons in constraints:
            vs = cons['vars']
            for i in range(len(vs)):
                for j in range(i + 1, len(vs)):
                    var_adj[vs[i]].add(vs[j])
                    var_adj[vs[j]].add(vs[i])

        visited = [False] * len(unknown_cells)
        components = []
        for v in range(len(unknown_cells)):
            if visited[v]:
                continue
            appears = any(v in cons['vars'] for cons in constraints)
            if not appears:
                visited[v] = True
                continue
            q = deque([v])
            comp = []
            visited[v] = True
            while q:
                cur = q.popleft()
                comp.append(cur)
                for nb in var_adj[cur]:
                    if not visited[nb]:
                        visited[nb] = True
                        q.append(nb)
            components.append(comp)

        definite_mines = set()
        for comp_vars in components:
            local_index_of = {gv: i for i, gv in enumerate(comp_vars)}
            n = len(comp_vars)
            local_constraints = []
            for cons in constraints:
                if any(gv in local_index_of for gv in cons['vars']):
                    local_vars = [local_index_of[gv] for gv in cons['vars'] if gv in local_index_of]
                    local_constraints.append({'vars': local_vars, 'required': cons['required']})

            if any(c['required'] < 0 or c['required'] > len(c['vars']) for c in local_constraints):
                continue

            m = len(local_constraints)
            var_to_cons = [[] for _ in range(n)]
            for ci, cons in enumerate(local_constraints):
                for lv in cons['vars']:
                    var_to_cons[lv].append(ci)

            order_seq = sorted(range(n), key=lambda i: -len(var_to_cons[i]))
            assigned = [-1] * n
            assigned_sum = [0] * m
            unassigned_count = [len(local_constraints[i]['vars']) for i in range(m)]
            counts_true = [0] * n
            solutions_found = 0
            solution_limit = max_solutions

            def backtrack(pos):
                nonlocal solutions_found
                if solution_limit is not None and solutions_found >= solution_limit:
                    return
                if pos == n:
                    solutions_found += 1
                    for i in range(n):
                        if assigned[i] == 1:
                            counts_true[i] += 1
                    return

                var = order_seq[pos]
                for val in (0, 1):
                    assigned[var] = val
                    violated = False
                    modified = []
                    for ci in var_to_cons[var]:
                        prev_sum, prev_un = assigned_sum[ci], unassigned_count[ci]
                        assigned_sum[ci] += val
                        unassigned_count[ci] -= 1
                        modified.append((ci, prev_sum, prev_un))
                        req, s, rem = local_constraints[ci]['required'], assigned_sum[ci], unassigned_count[ci]
                        if s > req or s + rem < req:
                            violated = True
                            break
                    if not violated:
                        backtrack(pos + 1)
                    for ci, prev_sum, prev_un in modified:
                        assigned_sum[ci] = prev_sum
                        unassigned_count[ci] = prev_un
                    assigned[var] = -1
                    if solution_limit is not None and solutions_found >= solution_limit:
                        break

            backtrack(0)
            for local_i, gv in enumerate(comp_vars):
                if counts_true[local_i] == solutions_found:
                    definite_mines.add(unknown_cells[gv])

        return definite_mines

    def generate_minesweeper_puzzle(self):
        """生成扫雷谜题并隐藏一定比例的格子"""
        rows, cols = self.rows, self.cols
        min_p, max_p = self.unknown_range

        if rows == 9 and cols == 9:
            mines_range = (15, 30)
        elif rows == 12 and cols == 12:
            mines_range = (30, 45)
        else:
            mines_range = (max(5, rows*cols//10), min(30, rows*cols//3))

        mines_count = random.randint(*mines_range)
        grid = [[0]*cols for _ in range(rows)]
        cells = [(r,c) for r in range(rows) for c in range(cols)]
        random.shuffle(cells)

        for r,c in cells[:mines_count]:
            grid[r][c] = -1

        for r in range(rows):
            for c in range(cols):
                if grid[r][c] != -1:
                    grid[r][c] = sum(
                        1 for nr,nc in self.get_neighbors(r,c) if grid[nr][nc]==-1
                    )

        non_mine_cells = rows*cols - mines_count
        target_unknown_cells = int(rows*cols * random.uniform(min_p, max_p))
        hidden_non_mines_needed = max(0, target_unknown_cells - mines_count)
        hidden_non_mines_needed = min(hidden_non_mines_needed, non_mine_cells)

        non_mines = [(r,c) for r in range(rows) for c in range(cols) if grid[r][c]!=-1]
        random.shuffle(non_mines)
        hidden_non_mines = set(non_mines[:hidden_non_mines_needed])

        puzzle = []
        for r in range(rows):
            row = []
            for c in range(cols):
                if grid[r][c] == -1 or (r,c) in hidden_non_mines:
                    row.append(-2)
                else:
                    row.append(grid[r][c])
            puzzle.append(row)
        return puzzle

    def generate_and_check(self, puzzle, result_queue):
        try:
            solver_solution = self.find_definite_mines([row.copy() for row in puzzle])
            if not solver_solution:
                solver_result = "Unable to determine any mine locations."
            else:
                solver_result = sorted(list(solver_solution))
            result_queue.put({"success": True, "puzzle": puzzle, "solver_result": solver_result})
        except Exception as e:
            result_queue.put({"success": False, "error": str(e)})

    def save_parquet(self, filepath):
        df = pd.DataFrame(self.data)
        table = pa.Table.from_pandas(df)
        pq.write_table(table, filepath)
    def run(self, total=50, output_file="minesweeper.parquet"):
        while self.id_counter <= total:
            print(f"\n生成第 {self.id_counter} 个谜题...")
            result_queue = Queue()
            puzzle = self.generate_minesweeper_puzzle()
            p = multiprocessing.Process(target=self.generate_and_check, args=(puzzle, result_queue))
            p.start()
            try:
                result = result_queue.get(timeout=self.timeout)
                p.join()
                if not result["success"]:
                    print(f"生成失败: {result.get('error', '未知错误')}, 重新生成...")
                    continue
                solver_result = result["solver_result"]
                if (solver_result == "Unable to determine any mine locations." or
                    not (self.min_mines <= len(solver_result) <= self.max_mines)):
                    print("谜题不符合条件，重新生成...")
                    continue
                if self.use_letters:
                    puzzle_str = self.format_puzzle_as_letters(puzzle)
                    id_str = f"gen-03-letter--{self.difficulty}-{self.id_counter}"
                else:
                    puzzle_str = "\n".join(
                        " ".join(str(x) if x!=-2 else "." for x in row) for row in puzzle
                    )
                    id_str = f"gen-04-number--{self.difficulty}-{self.id_counter}"
                solution_bool = [[(r,c) in solver_result for c in range(self.cols)] for r in range(self.rows)]
                self.data.append({
                    "id": id_str,
                    "puzzle": puzzle_str,
                    "solution": solution_bool,
                    "no_adj": False,
                    "letter": self.use_letters,  
                    "regional": False
                })

                print(f"成功生成谜题 {self.id_counter}")
                self.id_counter += 1

            except Empty:
                p.terminate()
                p.join()
                print(f"生成超时，终止并重新生成...")
            except Exception as e:
                p.terminate()
                p.join()
                print(f"意外错误 {str(e)}, 重新生成...")

        self.save_parquet(output_file)
        print(f"\n所有谜题已保存到: {output_file}")

if __name__ == "__main__":
    generator = MinesweeperGenerator(
        rows=12, cols=12, unknown_range=(0.75,0.85),
        min_mines=5, max_mines=10, difficulty="hard",
        use_letters=True  
    )
    generator.run(total=50, output_file=r"C:\Users\ohhhh\Desktop\minesweeper_hard\minesweeper_puzzles_letters.parquet")
