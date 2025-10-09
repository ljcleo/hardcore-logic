from Hanoi_Gen01 import Hanoi_Gen01
import random
from collections import deque

'''
- task: Hanoi
- subtask: Custom disks order
- introduction: Restrictions on the order of the discs
'''

class Hanoi_Gen03(Hanoi_Gen01):
    def __init__(self, num_pegs, num_disks):
        self.num_pegs = num_pegs
        self.num_disks = num_disks
        super().__init__(self.num_pegs, self.num_disks)
        self.original_target=list(range(1,self.num_disks+1))
        self.new_target=self.random_list()
        self.initial_state = self.generate_random_initial_state()
        self.new_state = self.remap_initial_state()
        self.solution = self.solve_bfs()

    def random_list(self):
        original_list=list(range(1,self.num_disks+1))
        target_list=random.sample(original_list,self.num_disks)
        return target_list

    def remap_initial_state(self):
        # 创建映射字典：{original_disk: new_disk}
        mapping = {original: new for original, new in zip(self.original_target, self.new_target)}

        # 深拷贝 initial_state 避免直接修改
        new_state = {peg: disks.copy() for peg, disks in self.initial_state.items()}

        # 替换磁盘编号
        for peg in new_state:
            new_state[peg] = [mapping[disk] for disk in new_state[peg]]

        return new_state

    def solve_bfs(self):
        """用BFS找到最短路径"""
        queue = deque()
        queue.append((self.initial_state, []))
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

        return None  # 无解（合法初始状态必有解）

    def format_start_and_goal(self):
        """格式化 new_state 和 goal_state 输出"""
        # Start 用 new_state
        start_lines = ["Start (bottom -> top):"]
        for peg in self.pegs:
            disks = " ".join(str(d) for d in self.new_state[peg])
            start_lines.append(f"{peg} | {disks}")

        # Goal 用最后一根柱子放置 new_target，从 top->bottom
        goal_state = {peg: [] for peg in self.pegs}
        goal_state[self.pegs[-1]] = self.new_target[::-1]  # 列表从左到右对应 top->bottom，打印是 bottom->top
        goal_lines = ["Goal (bottom -> top):"]
        for peg in self.pegs:
            disks = " ".join(str(d) for d in goal_state[peg])
            goal_lines.append(f"{peg} | {disks}")

        return "\n".join(start_lines + goal_lines)

if __name__ == "__main__":
    hanoi_gen03 = Hanoi_Gen03(3,4)
    print(hanoi_gen03.new_state)
    print(hanoi_gen03.new_target)
    print(hanoi_gen03.format_start_and_goal())
    print(hanoi_gen03.solution)
