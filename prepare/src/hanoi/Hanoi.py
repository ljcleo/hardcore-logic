import pandas as pd
import json
from collections import deque
import os

'''
- task: Hanoi
- subtask: Original
- introduction: Initially, the disks are arranged in order on the first column, and then moved to the last column according to the rules.
'''

class Hanoi:
    def __init__(self):
        self.configs = [
            (3, 3), (3, 4), (3, 5), (3, 6),
            (4, 4), (4, 5), (4, 6),
            (5, 5), (5, 6),
            (6, 6)
        ]
        self.data = []
        self.id_counter = 1

    def generate_puzzle_string(self, initial_state, target_state, pegs):
        lines = ["Start (bottom -> top):"]

        # 起始状态
        for peg in pegs:
            disks = initial_state[peg]
            if disks:
                disks_str = " ".join(str(d) for d in reversed(disks))  # 底部到顶部
                lines.append(f"{peg} | {disks_str}")
            else:
                lines.append(f"{peg} |")

        lines.append("Goal (bottom -> top):")

        # 目标状态
        for peg in pegs:
            disks = target_state[peg]
            if disks:
                disks_str = " ".join(str(d) for d in reversed(disks))  # 底部到顶部
                lines.append(f"{peg} | {disks_str}")
            else:
                lines.append(f"{peg} |")

        return "\n".join(lines)

    def generate_classic_initial_state(self, num_pegs, num_disks):
        pegs = [chr(ord('A') + i) for i in range(num_pegs)]
        state = {peg: [] for peg in pegs}
        state[pegs[0]] = list(range(num_disks, 0, -1))  # 从大到小
        return state, pegs

    def generate_target_state(self, num_pegs, num_disks):
        pegs = [chr(ord('A') + i) for i in range(num_pegs)]
        state = {peg: [] for peg in pegs}
        state[pegs[-1]] = list(range(num_disks, 0, -1))  # 从大到小
        return state, pegs

    def solve_bfs(self, initial_state, target_peg, pegs):

        def state_to_key(state):
            return tuple(tuple(disks) for peg, disks in sorted(state.items()))

        def is_goal_state(state):
            return len(state[target_peg]) == sum(len(disks) for disks in state.values())

        def get_possible_moves(state):
            moves = []
            for src in pegs:
                if not state[src]:
                    continue
                top_disk = state[src][-1]
                for dest in pegs:
                    if src != dest and (not state[dest] or state[dest][-1] > top_disk):
                        moves.append((src, dest))
            return moves

        def apply_move(state, move):
            src, dest = move
            new_state = {peg: disks.copy() for peg, disks in state.items()}
            new_state[dest].append(new_state[src].pop())
            return new_state

        queue = deque()
        queue.append((initial_state, 0))
        visited = set()
        visited.add(state_to_key(initial_state))

        while queue:
            current_state, steps = queue.popleft()

            if is_goal_state(current_state):
                return steps

            for move in get_possible_moves(current_state):
                new_state = apply_move(current_state, move)
                state_key = state_to_key(new_state)

                if state_key not in visited:
                    visited.add(state_key)
                    queue.append((new_state, steps + 1))

        return -1

    def generate_data(self):
        for num_pegs, num_disks in self.configs:

            initial_state, pegs = self.generate_classic_initial_state(num_pegs, num_disks)
            target_state, _ = self.generate_target_state(num_pegs, num_disks)
            target_peg = pegs[-1]
            solution_length = self.solve_bfs(initial_state, target_peg, pegs)
            puzzle_str = self.generate_puzzle_string(initial_state, target_state, pegs)
            order = list(range(1, num_disks + 1))

            self.data.append({
                "id": self.id_counter,
                "puzzle": puzzle_str,
                "solution": solution_length,
                "order": json.dumps(order),
                "right_only": False
            })

            self.id_counter += 1

    def save_to_parquet(self, filename="hanoi_tower_problems.parquet"):
        df = pd.DataFrame(self.data)
        df.to_parquet(filename, index=False)



# 生成数据
if __name__ == "__main__":
    generator = Hanoi()
    generator.generate_data()
    generator.save_to_parquet("hanoi-cmp.parquet")

