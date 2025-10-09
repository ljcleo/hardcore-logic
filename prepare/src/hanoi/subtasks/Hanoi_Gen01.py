import random
from collections import deque

'''
- task: Hanoi
- subtask: random initial state
- introduction: The initial positions of the disks are not all on the first pillar, but are randomly distributed under the condition that there is a solution.
'''

class Hanoi_Gen01:
    def __init__(self, num_pegs=3, num_disks=3):
        self.num_pegs = num_pegs
        self.num_disks = num_disks
        self.pegs = [chr(ord('A') + i) for i in range(num_pegs)]
        self.target_peg = self.pegs[-1]
        self.initial_state = self.generate_random_initial_state()
        self.solution = self.solve_bfs()

    def generate_random_initial_state(self):
        disks = list(range(self.num_disks, 0, -1))
        state = {peg: [] for peg in self.pegs}

        for disk in disks:
            valid_pegs = [peg for peg in self.pegs
                          if not state[peg] or state[peg][-1] > disk]
            if not valid_pegs:
                return self.generate_random_initial_state()
            peg = random.choice(valid_pegs)
            state[peg].append(disk)
        return state

    def is_goal_state(self, state):
        return len(state[self.target_peg]) == self.num_disks

    def get_possible_moves(self, state):
        moves = []
        for src in self.pegs:
            if not state[src]:
                continue
            top_disk = state[src][-1]
            for dest in self.pegs:
                if src != dest and (not state[dest] or state[dest][-1] > top_disk):
                    moves.append((src, dest))
        return moves

    def apply_move(self, state, move):
        src, dest = move
        new_state = {peg: disks.copy() for peg, disks in state.items()}
        new_state[dest].append(new_state[src].pop())
        return new_state

    def state_to_key(self, state):
        return tuple(tuple(disks) for peg, disks in sorted(state.items()))

    def solve_bfs(self):
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

        return None

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

if __name__ == "__main__":
    hanoi = Hanoi_Gen01(num_pegs=4, num_disks=5)
    print(hanoi.format_start_and_goal())

    solution = hanoi.solve_bfs()
    print(solution)
    print(len(solution))