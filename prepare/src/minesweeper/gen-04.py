import random
from collections import deque
import itertools
import pandas as pd
from pysat.formula import CNF
from pysat.solvers import Glucose3

class RegionalMinesweeperGenerator:
    DIRS = [(-1,-1),(-1,0),(-1,1),
            (0,-1),        (0,1),
            (1,-1),(1,0),(1,1)]
    
    def __init__(self, mode='easy', count=50, use_letter=False):
        self.mode = mode
        self.count = count
        self.use_letter = use_letter
        self.config = self.get_config_by_mode(mode)
        if mode=='easy':
            self.solution_mine_range = (2, 5)
        elif mode=='medium':
            self.solution_mine_range = (3, 8)
        elif mode=='hard':
            self.solution_mine_range = (4, 10)
        else:
            raise ValueError("mode must be 'easy', 'medium' or 'hard'")

    def get_config_by_mode(self, mode):
        if mode=='easy':
            return {
                "grid_size": (9, 9),
                "mines_range": (6, 10),
                "reveal_ratio_range": (0.45, 0.5),
                "max_unknowns": 50,
                "trials_per_field": 20
            }
        elif mode=='medium':
            return {
                "grid_size": (9, 9),
                "mines_range": (8, 15),
                "reveal_ratio_range": (0.45, 0.5),
                "max_unknowns": 50,
                "trials_per_field": 20
            }
        elif mode=='hard':
            return {
                "grid_size": (12, 12),
                "mines_range": (7, 18),
                "reveal_ratio_range": (0.55, 0.6),
                "max_unknowns": 70,
                "trials_per_field": 20
            }
        else:
            raise ValueError("mode must be 'easy', 'medium' or 'hard'")

    def neighbors(self, r, c, R, C):
        for dr, dc in self.DIRS:
            nr, nc = r+dr, c+dc
            if 0 <= nr < R and 0 <= nc < C:
                yield nr, nc

    def compute_cluster_count_for_cell(self, mine_set, r, c, R, C):
        neigh = [(nr, nc) for nr, nc in self.neighbors(r, c, R, C) if (nr, nc) in mine_set]
        if not neigh:
            return 0
        seen = set()
        comps = 0
        pos_set = set(neigh)
        for p in neigh:
            if p in seen:
                continue
            comps += 1
            dq = deque([p])
            seen.add(p)
            while dq:
                x, y = dq.popleft()
                for nx, ny in self.neighbors(x, y, R, C):
                    if (nx, ny) in pos_set and (nx, ny) not in seen:
                        seen.add((nx, ny))
                        dq.append((nx, ny))
        return comps

    def compute_all_clues(self, mine_set, R, C):
        clues = [[0]*C for _ in range(R)]
        for r in range(R):
            for c in range(C):
                clues[r][c] = self.compute_cluster_count_for_cell(mine_set, r, c, R, C)
        return clues

    def generate_random_minefield(self, R, C, mines_count):
        cells = [(r,c) for r in range(R) for c in range(C)]
        mines = set(random.sample(cells, mines_count))
        return mines

    def make_puzzle_from_field(self, mine_set, R, C, reveal_non_mine_ratio=0.85, max_unknowns=20):
        total = R*C
        non_mines = [(r,c) for r in range(R) for c in range(C) if (r,c) not in mine_set]
        random.shuffle(non_mines)
        reveal_count = int(len(non_mines) * reveal_non_mine_ratio)
        while total - reveal_count > max_unknowns and reveal_count < len(non_mines):
            reveal_count += 1
        revealed = set(non_mines[:reveal_count])
        clues = self.compute_all_clues(mine_set, R, C)
        puzzle = [[-2]*C for _ in range(R)]
        for r in range(R):
            for c in range(C):
                if (r,c) in revealed:
                    puzzle[r][c] = clues[r][c]
                else:
                    puzzle[r][c] = -2
        unknown_count = sum(1 for r in range(R) for c in range(C) if puzzle[r][c] == -2)
        return puzzle, revealed, unknown_count
   
    def deduce_forced_mines_cluster_fast(self, puzzle, max_unknowns=None):
        #Regional Minesweeper Puzzle Solver (Based on SAT)
        R, C = len(puzzle), len(puzzle[0])
        unknowns = [(r,c) for r in range(R) for c in range(C) if puzzle[r][c] == -2]
        var_id = {p:i+1 for i,p in enumerate(unknowns)}  
        def var(p): return var_id[p]

        cnf = CNF()

        for r in range(R):
            for c in range(C):
                if puzzle[r][c] < 0:
                    continue
                neigh = [(nr, nc) for nr, nc in self.neighbors(r, c, R, C) if puzzle[nr][nc] == -2]
                if not neigh:
                    continue

                valid_patterns = []
                for bits in itertools.product([0, 1], repeat=len(neigh)):
                    mine_set = {neigh[i] for i, b in enumerate(bits) if b == 1}
                    if self.compute_cluster_count_for_cell(mine_set, r, c, R, C) == puzzle[r][c]:
                        valid_patterns.append(bits)

                if not valid_patterns:
                    continue

                for bits in itertools.product([0, 1], repeat=len(neigh)):
                    if bits in valid_patterns:
                        continue
                    clause = []
                    for (p, b) in zip(neigh, bits):
                        clause.append(-var(p) if b == 1 else var(p))
                    cnf.append(clause)

        solver = Glucose3()
        solver.append_formula(cnf)

        solutions = []
        while solver.solve():
            model = solver.get_model()
            mine_set = set(p for p in unknowns if var(p) in model)
            solutions.append(mine_set)

            blocking_clause = []
            model_set = set(model)
            for p in unknowns:
                v = var(p)
                if v in model_set:
                    blocking_clause.append(-v)
                else:
                    blocking_clause.append(v)
            solver.add_clause(blocking_clause)

        solver.delete()

        if not solutions:
            return set()

        definite = set(solutions[0])
        for sol in solutions[1:]:
            definite &= sol
        return definite

    def apply_letter_mapping(self, puzzle_str):
        mapping = {0:'Z', 1:'A', 2:'B', 3:'C', 4:'D', 5:'E', 6:'F', 7:'G', 8:'H'}
        lines = puzzle_str.split("\n")
        new_lines = []
        for line in lines:
            new_line = []
            for x in line.split():
                if x == '.':
                    new_line.append('.')
                else:
                    new_line.append(mapping[int(x)])
            new_lines.append(" ".join(new_line))
        return "\n".join(new_lines)

    def solution_to_json_str(self, solution):
        return "[" + ",".join(
        "[" + ",".join("true" if cell else "false" for cell in row) + "]"
        for row in solution
        ) + "]"


    def generate_single_puzzle(self, puzzle_id, max_attempts=100):
        R, C = self.config["grid_size"]
        attempts = 0
        while attempts < max_attempts:
            attempts += 1
            mines_cnt = random.randint(*self.config["mines_range"])
            mine_set = self.generate_random_minefield(R, C, mines_cnt)
            for _ in range(self.config["trials_per_field"]):
                reveal_ratio = random.uniform(*self.config["reveal_ratio_range"])
                puzzle, _, unknown_count = self.make_puzzle_from_field(
                    mine_set, R, C,
                    reveal_non_mine_ratio=reveal_ratio,
                    max_unknowns=self.config["max_unknowns"]
                )
                cluster_solution = self.deduce_forced_mines_cluster_fast(puzzle)
                if cluster_solution is None:
                    continue
                num_solution_mines = sum((r,c) in cluster_solution for r in range(R) for c in range(C))
                min_mines, max_mines = self.solution_mine_range
                if not (min_mines <= num_solution_mines <= max_mines):
                    continue
                solution_bool = [[(r,c) in cluster_solution for c in range(C)] for r in range(R)]
                puzzle_str = "\n".join(" ".join(str(x) if x != -2 else "." for x in row) for row in puzzle)
                if self.use_letter:
                    puzzle_str = self.apply_letter_mapping(puzzle_str)
                    letter_flag = True
                    puzzle_id = puzzle_id.replace("number", "letter")
                else:
                    letter_flag = False
                solution_str = self.solution_to_json_str(solution_bool)
                return {
                    "id": puzzle_id,
                    "puzzle": puzzle_str,
                    "solution": solution_str,
                    "no_adj": False,
                    "letter": letter_flag,
                    "regional": True
                }
        raise RuntimeError(f"Failed to generate puzzles that meet the required number of answer mines ({puzzle_id})")

    def generate_puzzles(self):
        puzzles = []
        for idx in range(1, self.count+1):
            puzzle_id = f"gen-04-number--{self.mode}-{idx:02d}"
            p = self.generate_single_puzzle(puzzle_id)
            puzzles.append(p)
        return puzzles

    def save_to_parquet(self, puzzles, file_path):
        df = pd.DataFrame(puzzles)
        df.to_parquet(file_path, index=False)

if __name__ == "__main__":
    gen = RegionalMinesweeperGenerator(mode="hard", count=5, use_letter=True)
    puzzles = gen.generate_puzzles()
    gen.save_to_parquet(puzzles, r"minesweeper_hard_letter.parquet")
