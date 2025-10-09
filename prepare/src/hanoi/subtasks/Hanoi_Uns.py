import random
from collections import deque

'''
- task: Hanoi
- subtask: Unsolvable puzzle
- introduction: Restrict the disk to move only to the right, thus constructing an unsolvable data set
'''
class Hanoi_Uns:
    def __init__(self, num_pegs=3, num_disks=3):
        self.num_pegs = num_pegs
        self.num_disks = num_disks
        self.pegs = [chr(ord('A') + i) for i in range(num_pegs)]
        self.target_peg = self.pegs[-1]  # 目标柱为最后一个柱子
        self.initial_state = self.generate_unsolvable_puzzle()

    def generate_random_initial_state(self):
        """生成随机合法初始状态"""
        disks = list(range(self.num_disks, 0, -1))  # 从大到小排列
        state = {peg: [] for peg in self.pegs}
        for disk in disks:
            valid_pegs = [peg for peg in self.pegs
                          if not state[peg] or state[peg][-1] > disk]
            peg = random.choice(valid_pegs) if valid_pegs else self.pegs[-1]
            state[peg].append(disk)
        return state

    def is_goal_state(self, state):
        """检查是否所有盘在目标柱"""
        return len(state[self.target_peg]) == self.num_disks

    def get_possible_moves(self, state):
        """生成合法移动（禁止回退）"""
        moves = []
        for src_idx in range(self.num_pegs):
            src = self.pegs[src_idx]
            if not state[src]:
                continue
            top_disk = state[src][-1]
            # 只能向右移动（src_idx < dest_idx）
            for dest_idx in range(src_idx + 1, self.num_pegs):
                dest = self.pegs[dest_idx]
                if not state[dest] or state[dest][-1] > top_disk:
                    moves.append((src, dest))
        return moves

    def apply_move(self, state, move):
        """执行移动并返回新状态"""
        src, dest = move
        new_state = {peg: disks.copy() for peg, disks in state.items()}
        new_state[dest].append(new_state[src].pop())
        return new_state

    def state_to_key(self, state):
        """状态转换为可哈希键"""
        return tuple(tuple(state[peg]) for peg in self.pegs)

    def solve_bfs(self):
        """BFS求解最短路径"""
        queue = deque([(self.initial_state, [])])
        visited = set()
        visited.add(self.state_to_key(self.initial_state))

        while queue:
            current_state, path = queue.popleft()
            if self.is_goal_state(current_state):
                return path
            for move in self.get_possible_moves(current_state):
                new_state = self.apply_move(current_state, move)
                state_key = self.state_to_key(new_state)
                if state_key not in visited:
                    visited.add(state_key)
                    queue.append((new_state, path + [move]))
        return None  # 无解

    def generate_unsolvable_puzzle(self, max_attempts=100):
        for _ in range(max_attempts):
            state = self.generate_random_initial_state()
            self.initial_state = state  # 更新当前初始状态
            solution = self.solve_bfs()
            if solution is None:
                return state
        raise RuntimeError(f"未能在 {max_attempts} 次尝试内生成无解问题")

    def format_state(self, state, title="State"):
        lines = [f"{title} (bottom -> top):"]
        for peg in self.pegs:
            disks = " ".join(str(d) for d in state[peg])
            lines.append(f"{peg} | {disks}")
        return "\n".join(lines)

    def format_start_and_goal(self):
        start_str = self.format_state(self.initial_state, "Start")
        goal_state = {peg: [] for peg in self.pegs}
        goal_state[self.target_peg] = list(range(self.num_disks, 0, -1))
        goal_str = self.format_state(goal_state, "Goal")
        return f"{start_str}\n{goal_str}"


# 使用示例
if __name__ == "__main__":
    # 创建4柱8盘的河内塔（目标柱自动为D）
    hanoi = Hanoi_Uns(num_pegs=3, num_disks=5)
    print("初始状态（目标柱:", hanoi.target_peg, "）：")
    print(hanoi.format_start_and_goal())
    print(hanoi.solve_bfs())

