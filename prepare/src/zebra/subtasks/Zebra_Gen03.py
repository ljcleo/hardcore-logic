from z3 import Solver, Int, Distinct, And, Or, Not, sat, Abs, unsat,Bool,Implies,Sum,If
import random
from copy import deepcopy
from Zebra_Gen02 import Zebra_Gen02

'''
- task : zebra
- subtask : add "or"/"imply" clue types
- introduction : In this section, we introduce "or" and "imply" clue types
to increase the coupling between constraints
'''

class Zebra_Gen03(Zebra_Gen02):
    def __init__(self, N=3, M=3, num=100,seed=None,num_fakes=-1):
        self.original_exactly_clues = []
        self.num_fakes = num_fakes
        super().__init__(N, M, num,seed)

        self.original_exactly_clues = self.generate_exactly_one_clues()
        self.global_constraints = self.generate_global_constraints()

        self.final_clues = self.simplify_clues()

    def generate_exactly_one_clues(self):
        """
        Construct pseudo-Boolean clues: each clue group has exactly one true and the rest are false.
        Each set contains 1 correct clue + 2 incorrect clues.
        """
        constraints = []
        used_values = {
            attr: list({row[attr] for row in self.original_solution})
            for attr in self.attributes
        }

        attempts = 0
        while (len(constraints) < self.num and attempts < self.num * 10):
            attempts += 1
            attr = random.choice(self.attributes)
            values = used_values[attr]

            true_clauses = []
            for val in values:
                var = self.z3_vars[attr][val]
                pos = next((i + 1 for i, row in enumerate(self.original_solution) if row[attr] == val), None)
                if pos:
                    true_clauses.append(var == pos)

            false_clauses = []
            for val in values:
                var = self.z3_vars[attr][val]
                true_pos = next((i + 1 for i, row in enumerate(self.original_solution) if row[attr] == val), None)
                all_wrong_pos = [i + 1 for i in range(self.N) if (i + 1) != true_pos]
                for wrong_pos in all_wrong_pos:
                    false_clauses.append(var == wrong_pos)

            if len(true_clauses) >= 1 and len(false_clauses) >= 2:
                chosen_true = random.choice(true_clauses)
                chosen_falses = random.sample(false_clauses, 2)
                all_conditions = [chosen_true] + chosen_falses
                random.shuffle(all_conditions)
                constraint = Sum([If(cond, 1, 0) for cond in all_conditions]) == 1
                constraints.append(constraint)

        return constraints

    def generate_global_constraints(self):
        constraints = []
        for attr in self.attributes:
            vars_list = []
            used_values = {row[attr] for row in self.original_solution}
            for value in used_values:
                var = self.z3_vars[attr][value]
                constraints.append(And(var >= 1, var <= self.N))
                vars_list.append(var)
            constraints.append(Distinct(vars_list))
        return constraints

    def simplify_clues(self):
        all_clues = []
        clues_by_type = {
            "FOUNDAT": self.original_foundat_clues,
            "SAMEHOUSE": self.original_samehouse_clues,
            "NOTAT": self.original_notat_clues,
            "DIRECTADJ": self.original_directadj_clues,
            "LEFTRIGHTOF": self.original_left_rightof_clues,
            "BETWEEN": self.original_between_clues,
            "SIDEBYSIDE":self.original_sidebyside_clues,
            "OR":self.original_or_clues,
            "IMPLICATE":self.original_imply_clues,
            "EXACTLY": self.original_exactly_clues
        }

        for clue_type, clues in clues_by_type.items():
            for clue in clues:
                all_clues.append({
                    "type": clue_type,
                    "expr": clue
                })

        type_weights = {
            "FOUNDAT": 0.99,
            "SAMEHOUSE": 1,
            "NOTAT": 1,
            "DIRECTADJ": 0.995,
            "LEFTRIGHTOF": 0.995,
            "BETWEEN": 0.99,
            "SIDEBYSIDE":0.99,
            "OR":0.98,
            "IMPLICATE":0.98,
            "EXACTLY":0.98,
        }

        current_clues = deepcopy(all_clues)

        def has_unique_solution(clue_exprs):
            solver = Solver()
            solver.add(self.global_constraints)
            solver.add(clue_exprs)

            if solver.check() != sat:
                return False

            must_match = []
            for i in range(self.N):
                for attr in self.attributes:
                    value = self.original_solution[i][attr]
                    var = self.z3_vars[attr][value]
                    must_match.append(var == i + 1)

            solver.push()
            solver.add(Not(And(must_match)))
            result = solver.check()
            solver.pop()

            return result == unsat

        random.shuffle(current_clues)
        i = 0
        while i < len(current_clues):
            candidate = current_clues[:i] + current_clues[i + 1:]

            clue_type = current_clues[i]["type"]
            weight = type_weights.get(clue_type, 1.0)
            if random.random() < weight:
                clue_exprs = [c["expr"] for c in candidate]
                if has_unique_solution(clue_exprs):
                    current_clues = candidate
                    continue
            i += 1

        simplified = [c["expr"] for c in current_clues]
        return simplified

if __name__ == "__main__":
    zebra_gen03 = Zebra_Gen03(3,4 , 100)
    print("===== original_solution =====")
    for i, h in enumerate(zebra_gen03.original_solution):
        print(f"House {i+1}: {h}")

    print("===== final Z3 Constraints =====")
    for i, c in enumerate(zebra_gen03.final_clues):
        print(f"{i+1}. {c}")

    print("===== check =====")
    zebra_gen03.verify_final_clues_solution_matches_original()




