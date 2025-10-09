import random
from collections import defaultdict, deque
import time
from tqdm import tqdm
from bitarray import bitarray
import pandas as pd
import pyarrow

class HitoriGenerator:
    """Hitori谜题生成器类，用于生成具有唯一解的Hitori谜题，支持数字到字母的加密功能"""
    
    def __init__(self, size, min_black=None, max_black=None, exact_black=None):
        """
        初始化Hitori生成器，根据谜题尺寸设置默认黑格数量参数
        
        参数:
            size: 谜题尺寸 (n x n)
            min_black: 黑格数量最小值（如未提供则使用默认值）
            max_black: 黑格数量最大值（如未提供则使用默认值）
            exact_black: 黑格数量精确值（如未提供则使用默认值，优先级高于范围）
        """
        self.size = size
        
        # 根据尺寸设置默认黑格数量参数
        self._set_default_black_counts()
        
        # 如果用户提供了参数，则覆盖默认值
        if min_black is not None:
            self.min_black = min_black
        if max_black is not None:
            self.max_black = max_black
        if exact_black is not None:
            self.exact_black = exact_black
        
        # 设置随机种子，确保一定的可重复性
        random.seed(42)
    
    def _set_default_black_counts(self):
        """根据谜题尺寸设置默认的黑格数量参数"""
        if self.size == 4:
            self.min_black = None
            self.max_black = None
            self.exact_black = 5  # 4x4默认黑格数为5
        elif self.size == 5:
            self.min_black = None
            self.max_black = None
            self.exact_black = 8  # 5x5默认黑格数为8
        elif self.size == 6:
            self.min_black = 10   # 6x6默认黑格范围10-11
            self.max_black = 11
            self.exact_black = None
        elif self.size == 7:
            self.min_black = 10   # 7x7默认黑格范围10-14
            self.max_black = 14
            self.exact_black = None
        else:
            # 对于其他尺寸，默认不限制黑格数量
            self.min_black = None
            self.max_black = None
            self.exact_black = None
    
    @staticmethod
    def create_mapping(size):
        """根据谜题大小创建数字到字母的映射规则"""
        # 生成足够的字母（A, B, C, ...）
        letters = [chr(ord('A') + i) for i in range(size)]
        
        # 创建映射：每行是一个字典，数字1-size映射到相应字母
        # 每行的字母依次循环移位
        mapping = []
        for row_idx in range(size):
            row_mapping = {}
            for num in range(1, size+1):
                # 计算当前数字对应的字母索引，实现每行移位效果
                letter_idx = (num - 1 + row_idx) % size
                row_mapping[num] = letters[letter_idx]
            mapping.append(row_mapping)
        return mapping
    
    def encrypt_puzzle(self, grid):
        """使用create_mapping生成的规则将数字谜题加密为字母谜题"""
        mapping = self.create_mapping(self.size)
        encrypted_grid = []
        for row_idx, row in enumerate(grid):
            encrypted_row = [mapping[row_idx][num] for num in row]
            encrypted_grid.append(encrypted_row)
        return encrypted_grid
    
    def random_grid(self):
        """生成随机棋盘，减少初始冲突以提高效率"""
        grid = []
        for _ in range(self.size):
            row = []
            # 生成行时尽量避免过多重复
            for _ in range(self.size):
                if row:
                    existing = set(row)
                    # 70%概率选择不重复的数字
                    if len(existing) < self.size and random.random() < 0.7:
                        candidates = [num for num in range(1, self.size+1) if num not in existing]
                        if candidates:
                            row.append(random.choice(candidates))
                            continue
                row.append(random.randint(1, self.size))
            grid.append(row)
        return grid
    
    def build_conflict_groups(self, grid):
        """同时找出行和列中重复值的分组，用于更有效的剪枝"""
        row_groups = []
        col_groups = []
        
        # 行重复组
        for r in range(self.size):
            pos_by_val = defaultdict(list)
            for c in range(self.size):
                pos_by_val[grid[r][c]].append((r, c))
            for val, poses in pos_by_val.items():
                if len(poses) > 1:
                    row_groups.append((val, poses, 'row'))
        
        # 列重复组
        for c in range(self.size):
            pos_by_val = defaultdict(list)
            for r in range(self.size):
                pos_by_val[grid[r][c]].append((r, c))
            for val, poses in pos_by_val.items():
                if len(poses) > 1:
                    col_groups.append((val, poses, 'col'))
        
        # 合并并按组大小排序（大组优先处理，提高剪枝效率）
        all_groups = row_groups + col_groups
        all_groups.sort(key=lambda x: -len(x[1]))
        return all_groups
    
    @staticmethod
    def is_adjacent(a, b):
        """检查两个单元格是否相邻"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
    
    def check_connectivity_fast(self, grid, black_mask):
        """快速检查白格是否连通，使用位运算优化"""
        # 计算白格总数
        white_count = self.size * self.size - black_mask.count()
        if white_count == 0:
            return False
        if white_count == 1:
            return True
        
        # 找到第一个白格作为起点
        start_idx = None
        for i in range(self.size * self.size):
            if not black_mask[i]:
                start_idx = i
                break
        if start_idx is None:
            return False
        
        # BFS检查连通性
        visited = bitarray(self.size * self.size)
        visited.setall(False)
        queue = deque([start_idx])
        visited[start_idx] = True
        visited_count = 1
        
        while queue:
            idx = queue.popleft()
            r, c = idx // self.size, idx % self.size
            
            # 检查四个方向
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    nidx = nr * self.size + nc
                    if not black_mask[nidx] and not visited[nidx]:
                        visited[nidx] = True
                        visited_count += 1
                        queue.append(nidx)
                        # 早期退出条件
                        if visited_count == white_count:
                            return True
        
        return visited_count == white_count
    
    def solve_hitori_optimized(self, grid, max_solutions=2, time_limit=5.0):
        """优化的求解器，使用位运算和更强的剪枝策略"""
        start_time = time.time()
        groups = self.build_conflict_groups(grid)
        
        # 使用位掩码表示黑格，比集合操作更快
        black_mask = bitarray(self.size * self.size)
        black_mask.setall(False)
        
        # 跟踪每行每列的白色单元格中的值
        row_white_values = [defaultdict(int) for _ in range(self.size)]
        col_white_values = [defaultdict(int) for _ in range(self.size)]
        
        # 跟踪单元格状态：None=未决定, True=白, False=黑
        cell_state = [[None for _ in range(self.size)] for _ in range(self.size)]
        
        solutions = []
        
        def index(r, c):
            """将行、列转换为位掩码索引"""
            return r * self.size + c
        
        def backtrack(group_idx):
            # 检查时间限制
            if time.time() - start_time > time_limit:
                return True
            # 检查是否已找到足够解
            if len(solutions) >= max_solutions:
                return True
            
            # 所有组处理完毕
            if group_idx == len(groups):
                # 处理未决定的单元格（设为白色）
                for r in range(self.size):
                    for c in range(self.size):
                        if cell_state[r][c] is None:
                            val = grid[r][c]
                            # 检查列冲突
                            if col_white_values[c].get(val, 0) >= 1:
                                return False
                            # 检查行冲突
                            if row_white_values[r].get(val, 0) >= 1:
                                return False
                            # 标记为白色
                            cell_state[r][c] = True
                            row_white_values[r][val] += 1
                            col_white_values[c][val] += 1
            
                # 检查连通性
                if self.check_connectivity_fast(grid, black_mask):
                    solutions.append(black_mask.copy())
            
                # 回溯未决定的单元格
                for r in range(self.size):
                    for c in range(self.size):
                        if cell_state[r][c] is True and not black_mask[index(r, c)]:
                            val = grid[r][c]
                            row_white_values[r][val] -= 1
                            col_white_values[c][val] -= 1
                            cell_state[r][c] = None
            
                return False
            
            # 处理当前组
            val, poses, group_type = groups[group_idx]
            
            # 尝试每组中保留一个白色，其余涂黑
            # 随机排序以增加找到解的概率
            shuffled_poses = list(poses)
            random.shuffle(shuffled_poses)
            
            # 选项1: 保留一个白色
            for keep_pos in shuffled_poses:
                r_keep, c_keep = keep_pos
                
                # 检查保留位置是否已被标记为黑色
                if cell_state[r_keep][c_keep] is False:
                    continue
                
                # 检查列中是否已有相同值的白色单元格
                if col_white_values[c_keep].get(val, 0) >= 1:
                    continue
                
                # 检查行中是否已有相同值的白色单元格
                if row_white_values[r_keep].get(val, 0) >= 1:
                    continue
                
                # 需要涂黑的位置
                to_black = [p for p in poses if p != keep_pos]
                
                # 检查涂黑位置是否有冲突
                valid = True
                for (r, c) in to_black:
                    # 检查是否已被标记为白色
                    if cell_state[r][c] is True:
                        valid = False
                        break
                    
                    # 检查与已有黑格是否相邻
                    idx = index(r, c)
                    # 上
                    if r > 0 and black_mask[index(r-1, c)]:
                        valid = False
                        break
                    # 下
                    if r < self.size-1 and black_mask[index(r+1, c)]:
                        valid = False
                        break
                    # 左
                    if c > 0 and black_mask[index(r, c-1)]:
                        valid = False
                        break
                    # 右
                    if c < self.size-1 and black_mask[index(r, c+1)]:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # 检查涂黑位置之间是否相邻
                for i in range(len(to_black)):
                    for j in range(i+1, len(to_black)):
                        if self.is_adjacent(to_black[i], to_black[j]):
                            valid = False
                            break
                    if not valid:
                        break
                
                if not valid:
                    continue
                
                # 保存状态用于回溯
                state_changes = []
                
                # 标记保留位置为白色
                if cell_state[r_keep][c_keep] is None:
                    cell_state[r_keep][c_keep] = True
                    row_white_values[r_keep][val] += 1
                    col_white_values[c_keep][val] += 1
                    state_changes.append(('keep', r_keep, c_keep, val))
                
                # 标记需要涂黑的位置
                for (r, c) in to_black:
                    if cell_state[r][c] is None:
                        cell_state[r][c] = False
                        black_mask[index(r, c)] = True
                        state_changes.append(('black', r, c))
                
                # 递归处理下一组
                stop = backtrack(group_idx + 1)
                if stop:
                    return True
                
                # 回溯
                for change in reversed(state_changes):
                    if change[0] == 'keep':
                        _, r, c, v = change
                        cell_state[r][c] = None
                        row_white_values[r][v] -= 1
                        col_white_values[c][v] -= 1
                    else:
                        _, r, c = change
                        cell_state[r][c] = None
                        black_mask[index(r, c)] = False
            
            # 选项2: 全部涂黑
            to_black_all = poses  # 整个冲突组都要涂黑
            valid_all = True
            
            # 检查涂黑位置是否有冲突
            for (r, c) in to_black_all:
                # 检查是否已被标记为白色
                if cell_state[r][c] is True:
                    valid_all = False
                    break
                
                # 检查与已有黑格是否相邻
                idx = index(r, c)
                if (r > 0 and black_mask[index(r-1, c)]) or \
                   (r < self.size-1 and black_mask[index(r+1, c)]) or \
                   (c > 0 and black_mask[index(r, c-1)]) or \
                   (c < self.size-1 and black_mask[index(r, c+1)]):
                    valid_all = False
                    break
            
            # 检查涂黑位置之间是否相邻
            if valid_all:
                for i in range(len(to_black_all)):
                    for j in range(i+1, len(to_black_all)):
                        if self.is_adjacent(to_black_all[i], to_black_all[j]):
                            valid_all = False
                            break
                    if not valid_all:
                        break
            
            if valid_all:
                state_changes = []
                for (r, c) in to_black_all:
                    if cell_state[r][c] is None:  # 只处理未决定的单元格
                        cell_state[r][c] = False
                        black_mask[index(r, c)] = True
                        state_changes.append(('black', r, c))
                
                # 递归处理下一组
                stop = backtrack(group_idx + 1)
                if stop:
                    return True
                
                # 回溯
                for change in reversed(state_changes):
                    _, r, c = change
                    cell_state[r][c] = None
                    black_mask[index(r, c)] = False
            
            return False
        
        backtrack(0)
        
        # 转换为坐标集合
        result = []
        seen = set()
        for sol in solutions:
            key = sol.tobytes()
            if key not in seen:
                seen.add(key)
                black_set = set()
                for i in range(self.size * self.size):
                    if sol[i]:
                        r, c = i // self.size, i % self.size
                        black_set.add((r, c))
                result.append(black_set)
        
        return result
    
    def generate_unique_hitori(self, attempts_limit=2000, time_limit_per_try=10.0):
        """生成唯一解并满足黑格数量条件的谜题"""
        for attempt in range(1, attempts_limit + 1):
            # 生成初始网格
            grid = self.random_grid()
            
            # 求解
            sols = self.solve_hitori_optimized(grid, max_solutions=2, time_limit=time_limit_per_try)
            
            # 检查是否有唯一解
            if len(sols) == 1:
                black_count = len(sols[0])
                
                # 检查黑格数量条件
                if self.exact_black is not None and black_count != self.exact_black:
                    continue
                if self.min_black is not None and black_count < self.min_black:
                    continue
                if self.max_black is not None and black_count > self.max_black:
                    continue
                    
                return grid, sols[0]
        
        return None, None
    
    def format_puzzle(self, grid, encrypted=False):
        """
        将grid转换为指定格式：每行开头1个空格，数字间3个空格，行间换行
        如果encrypted为True，则使用字母替换数字
        """
        # 如果需要加密，先转换为字母网格
        if encrypted:
            display_grid = self.encrypt_puzzle(grid)
        else:
            display_grid = grid
            
        return "\n".join([f" { '   '.join(map(str, row)) }" for row in display_grid])
    
    def format_solution(self, black_set):
        """将黑格集合转换为二维bool数组字符串（小写true/false）"""
        # 初始化全为false的二维数组
        bool_grid = [[False for _ in range(self.size)] for _ in range(self.size)]
        # 黑格位置设为true
        for (r, c) in black_set:
            bool_grid[r][c] = True
        # 转换为字符串（替换Python的True/False为小写）
        return str(bool_grid).replace("True", "true").replace("False", "false")
    
    def generate_batch(self, num_puzzles, output_file=None, attempts_limit=10000, 
                      time_limit_per_try=30.0, encrypted=False):
        """
        批量生成谜题并保存为Parquet格式
        
        参数:
            num_puzzles: 要生成的谜题数量
            output_file: 输出文件路径，默认为"{size}x{size}_hitori_puzzles.parquet"
            attempts_limit: 每个谜题的最大尝试次数
            time_limit_per_try: 每个谜题的求解时间限制
            encrypted: 是否对谜题进行加密（数字转字母）
            
        返回:
            生成的谜题数据列表和保存的文件路径
        """
        # 存储谜题数据的列表（用于构建DataFrame）
        puzzle_data = []
        
        # 进度条
        with tqdm(total=num_puzzles, desc=f"生成{self.size}x{self.size} Hitori谜题") as pbar:
            puzzle_idx = 1  # 谜题序号（用于id）
            while len(puzzle_data) < num_puzzles:
                # 生成唯一解谜题
                grid, black_set = self.generate_unique_hitori(
                    attempts_limit=attempts_limit,
                    time_limit_per_try=time_limit_per_try
                )
                
                if grid and black_set:
                    # 1. 生成id：gen-04--{size}x{size}-{idx}
                    puzzle_id = f"gen-04--{self.size}x{self.size}-{puzzle_idx}"
                    # 2. 格式化puzzle，根据encrypted参数决定是否加密
                    formatted_puzzle = self.format_puzzle(grid, encrypted)
                    # 3. 格式化solution（二维bool数组）
                    formatted_solution = self.format_solution(black_set)
                    # 4. 记录加密状态
                    encrypted_status = encrypted
                    
                    # 添加到数据列表
                    puzzle_data.append({
                        "id": puzzle_id,
                        "puzzle": formatted_puzzle,
                        "solution": formatted_solution,
                        "encrypted": encrypted_status
                    })
                    
                    puzzle_idx += 1
                    pbar.update(1)
        
        # 确定输出文件路径
        if output_file is None:
            # 根据是否加密添加不同的后缀
            encryption_suffix = "_encrypted" if encrypted else ""
            output_file = r"C:\Users\ohhhh\Desktop\{self.size}x{self.size}_hitori_cmp-01-1.parquet"

        # 转换为DataFrame并保存为Parquet
        df = pd.DataFrame(puzzle_data)
        df.to_parquet(output_file, engine="pyarrow", index=False)
        
        return puzzle_data, output_file


# 示例用法
if __name__ == "__main__":
    # 示例1: 生成4x4谜题（默认黑格数为5）
    print("生成4x4谜题示例:")
    generator_4x4 = HitoriGenerator(size=4)
    puzzles_4x4, path_4x4 = generator_4x4.generate_batch(
        num_puzzles=2,
        time_limit_per_try=30.0,
        encrypted=False
    )
    print(f"4x4谜题已保存至: {path_4x4}\n")
    
    # 示例2: 生成6x6谜题并加密（默认黑格范围10-11）
    print("生成6x6加密谜题示例:")
    '''
    generator_6x6 = HitoriGenerator(size=6)
    puzzles_6x6, path_6x6 = generator_6x6.generate_batch(
        num_puzzles=2,
        time_limit_per_try=30.0,
        encrypted=True  # 启用加密
    )
    print(f"6x6加密谜题已保存至: {path_6x6}")
    
    # 打印一个加密谜题示例
    if puzzles_6x6:
        print(f"\n加密谜题示例:")
        print(f"id: {puzzles_6x6[0]['id']}")
        print(f"puzzle:\n{puzzles_6x6[0]['puzzle']}")
        print(f"solution: {puzzles_6x6[0]['solution']}")
        print(f"encrypted: {puzzles_6x6[0]['encrypted']}")
    '''