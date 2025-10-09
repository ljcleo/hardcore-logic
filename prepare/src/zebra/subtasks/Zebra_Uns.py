from z3 import Solver, Int, Distinct, And, Or, Not, sat, Abs, unsat,Bool,Implies,If,Sum
import random
from Zebra_Gen03 import Zebra_Gen03

'''
- task : zebra
- subtask : unsolvable
- introduction : In this section, add constraints that contradict the initial solution to construct an unsolvable problem
'''

class Zebra_Unsolvable(Zebra_Gen03):
    def __init__(self, N=3, M=3, num=10, seed=None, num_fakes=3):
        super().__init__(N, M, num, seed)

        self.fake_clues = self.generate_fake_clues(num_fakes=num_fakes)
        self.unsolvable_clues = self.final_clues + self.fake_clues

        self.is_unsolvable = self.verify_unsolvability(self.unsolvable_clues)

    def generate_fake_clues(self, num_fakes=3):
        fake_clues = []
        generators = [
            self.inject_false_implication,
            self.inject_false_adjacent,
            self.inject_false_samehouse,
            self.inject_false_not_samehouse,
            self.inject_false_foundat
        ]
        for _ in range(num_fakes):
            func = random.choice(generators)
            clue = func()
            if clue is not None:
                fake_clues.append(clue)
        return fake_clues

    def inject_false_implication(self):
        value_positions = {
            self.z3_vars[attr][row[attr]]: i + 1
            for i, row in enumerate(self.original_solution)
            for attr in self.attributes
        }
        for _ in range(20):
            attr_a = random.choice(self.attributes)
            val_a = random.choice(list(self.z3_vars[attr_a].keys()))
            var_a = self.z3_vars[attr_a][val_a]
            pos_a = value_positions[var_a]
            clause_a = var_a == pos_a

            attr_b = random.choice([a for a in self.attributes if a != attr_a])
            val_b = random.choice(list(self.z3_vars[attr_b].keys()))
            var_b = self.z3_vars[attr_b][val_b]
            pos_b = value_positions[var_b]

            fake_pos = random.choice([p for p in range(1, self.N + 1) if p != pos_b])
            clause_b = var_b == fake_pos

            return Implies(clause_a, clause_b)

    def inject_false_adjacent(self):
        for attr1 in self.attributes:
            for attr2 in self.attributes:
                if attr1 == attr2:
                    continue
                for row in self.original_solution:
                    val1 = row[attr1]
                    val2 = row[attr2]
                    pos1 = next(i for i, r in enumerate(self.original_solution) if r[attr1] == val1)
                    pos2 = next(i for i, r in enumerate(self.original_solution) if r[attr2] == val2)
                    if abs(pos1 - pos2) > 1:
                        var1 = self.z3_vars[attr1][val1]
                        var2 = self.z3_vars[attr2][val2]
                        return Abs(var1 - var2) == 1
        return None

    def inject_false_samehouse(self):
        for _ in range(20):
            a1, a2 = random.sample(self.attributes, 2)
            val1 = random.choice(list(self.z3_vars[a1].keys()))
            val2 = random.choice(list(self.z3_vars[a2].keys()))
            pos1 = next(i for i, r in enumerate(self.original_solution) if r[a1] == val1)
            pos2 = next(i for i, r in enumerate(self.original_solution) if r[a2] == val2)
            if pos1 != pos2:
                var1 = self.z3_vars[a1][val1]
                var2 = self.z3_vars[a2][val2]
                return var1 == var2
        return None

    def inject_false_not_samehouse(self):
        for _ in range(20):
            a1, a2 = random.sample(self.attributes, 2)
            val1 = random.choice(list(self.z3_vars[a1].keys()))
            val2 = random.choice(list(self.z3_vars[a2].keys()))
            pos1 = next(i for i, r in enumerate(self.original_solution) if r[a1] == val1)
            pos2 = next(i for i, r in enumerate(self.original_solution) if r[a2] == val2)
            if pos1 == pos2:
                var1 = self.z3_vars[a1][val1]
                var2 = self.z3_vars[a2][val2]
                return var1 != var2
        return None

    def inject_false_foundat(self):
        for attr in self.attributes:
            for val in self.z3_vars[attr]:
                var = self.z3_vars[attr][val]
                correct_pos = next(i + 1 for i, r in enumerate(self.original_solution) if r[attr] == val)
                wrong_positions = [i for i in range(1, self.N + 1) if i != correct_pos]
                if wrong_positions:
                    wrong_pos = random.choice(wrong_positions)
                    return var == wrong_pos
        return None

    def verify_unsolvability(self, clues):
        solver = Solver()
        solver.add(self.global_constraints)
        solver.add(clues)
        return solver.check() == unsat

if __name__ == '__main__':
    zebra_uns = Zebra_Unsolvable(3, 3, 50, 2)

    print("====Original solution====")
    for i, h in enumerate(zebra_uns.original_solution):
        print(f"House {i+1}: {h}")

    print("===== All Combined Clues (final + fake) =====")
    for i, clue in enumerate(zebra_uns.unsolvable_clues):
        prefix = "F" if i < len(zebra_uns.final_clues) else "X"
        print(f"[{prefix}{i+1}] {clue}")

