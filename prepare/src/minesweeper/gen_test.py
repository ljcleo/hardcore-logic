import random
import csv
import time
from collections import deque
from tqdm import tqdm

# ----------------- 区域扫雷核心函数 -----------------
DIRS = [(-1,-1),(-1,0),(-1,1),
        (0,-1),        (0,1),
        (1,-1),(1,0),(1,1)]

def neighbors(r, c, R, C):
    for dr, dc in DIRS:
        nr, nc = r+dr, c+dc
        if 0 <= nr < R and 0 <= nc < C:
            yield nr, nc

def compute_cluster_count_for_cell(mine_set, r, c, R, C):
    neigh = [(nr, nc) for nr, nc in neighbors(r, c, R, C) if (nr, nc) in mine_set]
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
            for nx, ny in neighbors(x, y, R, C):
                if (nx, ny) in pos_set and (nx, ny) not in seen:
                    seen.add((nx, ny))
                    dq.append((nx, ny))
    return comps

def compute_all_clues(mine_set, R, C):
    clues = [[0]*C for _ in range(R)]
    for r in range(R):
        for c in range(C):
            clues[r][c] = compute_cluster_count_for_cell(mine_set, r, c, R, C)
    return clues

def generate_random_minefield(R, C, mines_count):
    cells = [(r,c) for r in range(R) for c in range(C)]
    mines = set(random.sample(cells, mines_count))
    return mines

def make_puzzle_from_field(mine_set, R, C, reveal_non_mine_ratio=0.85, max_unknowns=20):
    total = R*C
    non_mines = [(r,c) for r in range(R) for c in range(C) if (r,c) not in mine_set]
    random.shuffle(non_mines)
    reveal_count = int(len(non_mines) * reveal_non_mine_ratio)
    while total - reveal_count > max_unknowns and reveal_count < len(non_mines):
        reveal_count += 1
    revealed = set(non_mines[:reveal_count])
    clues = compute_all_clues(mine_set, R, C)
    puzzle = [[-2]*C for _ in range(R)]
    for r in range(R):
        for c in range(C):
            if (r,c) in revealed:
                puzzle[r][c] = clues[r][c]
            else:
                puzzle[r][c] = -2
    unknown_count = sum(1 for r in range(R) for c in range(C) if puzzle[r][c] == -2)
    return puzzle, revealed, unknown_count

# ----------------- 区域扫雷解法（优化版：增量交集） -----------------
def deduce_forced_mines_cluster_fast(puzzle, max_unknowns=60):
    R, C = len(puzzle), len(puzzle[0])
    unknowns = [(r,c) for r in range(R) for c in range(C) if puzzle[r][c]==-2]
    assignment = {}

    def propagate():
        changed = True
        while changed:
            changed = False
            for r in range(R):
                for c in range(C):
                    clue = puzzle[r][c]
                    if clue < 0: 
                        continue
                    neigh = list(neighbors(r, c, R, C))
                    unknown_neigh = [p for p in neigh if p not in assignment and puzzle[p[0]][p[1]]==-2]
                    mine_count = sum(assignment.get(p,0) for p in neigh)
                    if clue - mine_count == len(unknown_neigh):
                        for p in unknown_neigh:
                            if assignment.get(p,-1)!=1:
                                assignment[p] = 1
                                changed = True
                    elif clue - mine_count == 0:
                        for p in unknown_neigh:
                            if assignment.get(p,-1)!=0:
                                assignment[p] = 0
                                changed = True
        return
    propagate()

    remaining = [p for p in unknowns if p not in assignment]
    if len(remaining) > max_unknowns:
        return None

    definite_mines = None
    found_solution = False

    def backtrack(i=0):
        nonlocal definite_mines, found_solution
        if i == len(remaining):
            mine_set = {p for p, v in assignment.items() if v == 1}
            valid = True
            for r in range(R):
                for c in range(C):
                    if puzzle[r][c] >= 0 and compute_cluster_count_for_cell(mine_set, r, c, R, C) != puzzle[r][c]:
                        valid = False
                        break
                if not valid:
                    return
            found_solution = True
            if definite_mines is None:
                definite_mines = set(mine_set)
            else:
                definite_mines &= mine_set
            return

        if i > 0:
            current_p = remaining[i-1]
            for (r, c) in neighbors(current_p[0], current_p[1], R, C):
                if puzzle[r][c] < 0:
                    continue
                current_mine_set = {p for p, v in assignment.items() if v == 1}
                current_cluster = compute_cluster_count_for_cell(current_mine_set, r, c, R, C)
                if current_cluster > puzzle[r][c]:
                    return
        
        p = remaining[i]
        for v in (0, 1):
            assignment[p] = v
            backtrack(i + 1)
            del assignment[p]

    backtrack(0)
    if not found_solution:
        return None
    return definite_mines if definite_mines else set()

# ----------------- 普通扫雷解法 -----------------
def get_neighbors_standard(r, c, rows, cols):
    for dr in (-1,0,1):
        for dc in (-1,0,1):
            if dr==0 and dc==0:
                continue
            nr, nc = r+dr, c+dc
            if 0<=nr<rows and 0<=nc<cols:
                yield (nr,nc)

def find_definite_mines_standard(grid):
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    known_flags = set()
    unknown_cells = []
    coord_to_var = {}
    for r in range(rows):
        for c in range(cols):
            if grid[r][c]==-1:
                known_flags.add((r,c))
            elif grid[r][c]==-2:
                idx = len(unknown_cells)
                unknown_cells.append((r,c))
                coord_to_var[(r,c)] = idx

    constraints = []
    for r in range(rows):
        for c in range(cols):
            val = grid[r][c]
            if isinstance(val,int) and 0<=val<=8:
                adj_unknowns=[]
                adj_known_flags=0
                for nr,nc in get_neighbors_standard(r,c,rows,cols):
                    if (nr,nc) in coord_to_var:
                        adj_unknowns.append(coord_to_var[(nr,nc)])
                    elif (nr,nc) in known_flags:
                        adj_known_flags+=1
                if adj_unknowns:
                    required = val - adj_known_flags
                    constraints.append({'cell':(r,c),'vars':adj_unknowns,'required':required})

    if not constraints:
        return set()

    var_adj=[set() for _ in range(len(unknown_cells))]
    for cons in constraints:
        vs = cons['vars']
        for i in range(len(vs)):
            for j in range(i+1,len(vs)):
                var_adj[vs[i]].add(vs[j])
                var_adj[vs[j]].add(vs[i])

    visited=[False]*len(unknown_cells)
    components=[]
    for v in range(len(unknown_cells)):
        if visited[v]:
            continue
        appears=False
        for cons in constraints:
            if v in cons['vars']:
                appears=True
                break
        if not appears:
            visited[v]=True
            continue
        q=deque([v])
        comp=[]
        visited[v]=True
        while q:
            cur=q.popleft()
            comp.append(cur)
            for nb in var_adj[cur]:
                if not visited[nb]:
                    visited[nb]=True
                    q.append(nb)
        components.append(comp)

    definite_mines=set()
    for comp_vars in components:
        local_index_of={gv:i for i,gv in enumerate(comp_vars)}
        n=len(comp_vars)
        local_constraints=[]
        for cons in constraints:
            touch=False
            for gv in cons['vars']:
                if gv in local_index_of:
                    touch=True
                    break
            if touch:
                local_vars=[local_index_of[gv] for gv in cons['vars'] if gv in local_index_of]
                local_constraints.append({'vars':local_vars,'required':cons['required']})
        inconsistent=False
        for cons in local_constraints:
            if cons['required']<0 or cons['required']>len(cons['vars']):
                inconsistent=True
                break
        if inconsistent:
            continue

        m=len(local_constraints)
        var_to_cons=[[] for _ in range(n)]
        for ci,cons in enumerate(local_constraints):
            for lv in cons['vars']:
                var_to_cons[lv].append(ci)

        order_seq=sorted(range(n),key=lambda i:-len(var_to_cons[i]))
        assigned=[-1]*n
        assigned_sum=[0]*m
        unassigned_count=[len(local_constraints[i]['vars']) for i in range(m)]
        counts_true=[0]*n
        solutions_found=0

        def backtrack(pos):
            nonlocal solutions_found
            if pos==n:
                solutions_found+=1
                for i in range(n):
                    if assigned[i]==1:
                        counts_true[i]+=1
                return
            var=order_seq[pos]
            for val in (0,1):
                assigned[var]=val
                violated=False
                modified=[]
                for ci in var_to_cons[var]:
                    prev_sum=assigned_sum[ci]
                    prev_un=unassigned_count[ci]
                    assigned_sum[ci]+=val
                    unassigned_count[ci]-=1
                    modified.append((ci,prev_sum,prev_un))
                    req=local_constraints[ci]['required']
                    s=assigned_sum[ci]
                    rem=unassigned_count[ci]
                    if s>req or s+rem<req:
                        violated=True
                        break
                if not violated:
                    backtrack(pos+1)
                for ci,prev_sum,prev_un in modified:
                    assigned_sum[ci]=prev_sum
                    unassigned_count[ci]=prev_un
                assigned[var]=-1
        backtrack(0)
        if solutions_found==0:
            continue
        for local_i,gv in enumerate(comp_vars):
            if counts_true[local_i]==solutions_found:
                definite_mines.add(unknown_cells[gv])
    return definite_mines

# ----------------- 生成和比较谜题 -----------------
def generate_single_puzzle(config, timeout=600):
    start_time = time.time()
    R, C = config["grid_size"]
    if (R, C) == (9, 9):
        solve_max_unknowns = 35
    elif (R, C) == (12, 12):
        solve_max_unknowns = 55
    else:
        solve_max_unknowns = 30
    pre_check_threshold = solve_max_unknowns + 10
    
    while True:
        if time.time() - start_time > timeout:
            return None, "超时"
        mines_cnt = random.randint(*config["mines_range"])
        mine_set = generate_random_minefield(R, C, mines_cnt)
        for _ in range(config["trials_per_field"]):
            if time.time() - start_time > timeout:
                return None, "超时"
            reveal_ratio = random.uniform(*config["reveal_ratio_range"])
            puzzle, _, unknown_count = make_puzzle_from_field(
                mine_set, R, C,
                reveal_non_mine_ratio=reveal_ratio,
                max_unknowns=config["max_unknowns"]
            )
            if unknown_count > pre_check_threshold:
                continue
            cluster_solution = deduce_forced_mines_cluster_fast(
                puzzle, max_unknowns=solve_max_unknowns
            )
            if time.time() - start_time > timeout or cluster_solution is None:
                continue
            standard_solution = find_definite_mines_standard(puzzle)
            if time.time() - start_time > timeout or standard_solution is None:
                continue
            if set(cluster_solution) != set(standard_solution):
                return {
                    "puzzle": puzzle,
                    "standard_solution": sorted(list(standard_solution)),
                    "cluster_solution": sorted(list(cluster_solution))
                }, "成功"
    return None, "未找到符合条件的谜题"

def generate_and_compare_puzzles_with_config(config, target_count=20, timeout=600):
    puzzles = []
    attempts = 0
    timeouts = 0
    with tqdm(total=target_count, desc=f"生成谜题 (配置: {config['name']})") as pbar:
        while len(puzzles) < target_count:
            attempts += 1
            puzzle, status = generate_single_puzzle(config, timeout)
            if status == "超时":
                timeouts += 1
                if timeouts % 5 == 0:
                    print(f"\n警告: 已超时 {timeouts} 次，可能配置仍需调整")
                continue
            if puzzle:
                puzzle["id"] = len(puzzles) + 1
                puzzle["config"] = config["name"]
                puzzles.append(puzzle)
                pbar.update(1)
            if attempts > target_count * 200:
                print(f"\n尝试次数过多 ({attempts} 次)，停止生成。已找到: {len(puzzles)}")
                break
    return puzzles

# ----------------- 保存结果 -----------------
def puzzle_to_string(puzzle):
    return "\n".join(",".join(map(str, row)) for row in puzzle)

def solution_to_string(solution):
    return ";".join([f"({r},{c})" for r, c in solution])

def save_comparison_results(puzzles, path):
    try:
        with open(path,"w",newline="",encoding="utf-8") as f:
            writer=csv.writer(f)
            writer.writerow(["ID","配置名称","谜题数据","普通扫雷规则下的解","区域性扫雷规则下的解"])
            for p in puzzles:
                writer.writerow([
                    p["id"],
                    p["config"],
                    puzzle_to_string(p["puzzle"]),
                    solution_to_string(p["standard_solution"]),
                    solution_to_string(p["cluster_solution"])
                ])
        return True
    except Exception as e:
        print(f"保存文件时出错: {str(e)}")
        return False

# ----------------- 主程序 -----------------
if __name__=="__main__":
    configs = [
        {
            "name": "小型网格-低难度",
            "target_count": 20,
            "grid_size": (9, 9),
            "mines_range": (6, 10),
            "reveal_ratio_range": (0.45, 0.5),  # 12x12 0.5-0.6  9x9 0.3-0.45
            "max_unknowns": 50,
            "trials_per_field": 10
        }
    ]
    TIMEOUT = 300
    OUTPUT_FILE = r"C:\Users\ohhhh\Desktop\minesweeper_test_9x9_abcd.csv"
    print("开始生成并比较谜题...")
    all_puzzles = []
    for config in configs:
        print(f"\n===== 开始处理配置: {config['name']} =====")
        puzzles = generate_and_compare_puzzles_with_config(
            config, 
            target_count=config["target_count"],
            timeout=TIMEOUT
        )
        all_puzzles.extend(puzzles)
        print(f"配置 {config['name']} 处理完成，生成 {len(puzzles)}/{config['target_count']} 个谜题")
    if save_comparison_results(all_puzzles, OUTPUT_FILE):
        print(f"\n所有配置处理完成，共保存 {len(all_puzzles)} 个谜题到 {OUTPUT_FILE}")
    else:
        print(f"\n生成完成，但保存文件失败。共生成 {len(all_puzzles)} 个谜题")
