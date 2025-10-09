from z3 import Solver, Int, Distinct, And, Or, Not, sat, Abs, unsat,Bool,Implies,Sum
import random
from copy import deepcopy
from Zebra_Gen01 import Zebra_Gen01

'''
- task : zebra
- subtask : add "or"/"imply" clue types
- introduction : In this section, we introduce "or" and "imply" clue types
to increase the coupling between constraints
'''

class Zebra_Gen02(Zebra_Gen01):
    def __init__(self, N=3, M=3, num=100,seed=None, num_fakes=-1):
        self.original_or_clues = []
        self.original_imply_clues = []
        self.num = num
        self.num_fakes = num_fakes
        super().__init__(N, M, seed)

        self.original_or_clues = self.generate_or_clues()
        self.original_imply_clues = self.generate_implication_clues()
        self.global_constraints = self.generate_global_constraints()

        self.final_clues = self.simplify_clues()

    def generate_or_clues(self):
        """
        Automatically generate as many true and false OR type clues as possible.
        """
        constraints = []
        used_values_by_attr = {
            attr: list({row[attr] for row in self.original_solution})
            for attr in self.attributes
        }

        count = 0
        for attr_true in self.attributes:
            for val_true in used_values_by_attr[attr_true]:
                var_true = self.z3_vars[attr_true][val_true]
                true_pos = next(i + 1 for i, row in enumerate(self.original_solution) if row[attr_true] == val_true)
                true_clause = (var_true == true_pos)

                for attr_false in self.attributes:
                    for val_false in used_values_by_attr[attr_false]:
                        if attr_false == attr_true and val_false == val_true:
                            continue

                        wrong_positions = [i + 1 for i, row in enumerate(self.original_solution) if
                                           row[attr_false] != val_false]
                        if not wrong_positions:
                            continue

                        wrong_pos = random.choice(wrong_positions)
                        var_false = self.z3_vars[attr_false][val_false]
                        false_clause = (var_false == wrong_pos)

                        if random.random() < 0.5:
                            clause = Or(true_clause, false_clause)
                        else:
                            clause = Or(false_clause, true_clause)

                        constraints.append(clause)
                        count += 1
                        if count >= self.num:
                            return constraints

        return constraints

    def generate_implication_clues(self, false_a_ratio=0.5):
        """
        The implication under strict logic implies a clue generator:
        - A clause is var == pos (true or false)
        - If A is true, clause B must also be true in the original solution
        - The B clause is limited to simple logic: adjacent, greater than, not equal
        """
        constraints = []
        value_positions = {
            self.z3_vars[attr][row[attr]]: i + 1
            for i, row in enumerate(self.original_solution)
            for attr in self.attributes
        }

        def is_clause_true_in_solution(clause):
            s = Solver()
            for var, pos in value_positions.items():
                s.add(var == pos)
            s.add(Not(clause))
            return s.check() == unsat

        tries = 0
        max_tries = self.num * 10

        while len(constraints) < self.num and tries < max_tries:
            tries += 1

            # 构造 A 子句
            attr_a = random.choice(self.attributes)
            val_a = random.choice(list(self.z3_vars[attr_a].keys()))
            var_a = self.z3_vars[attr_a][val_a]
            true_pos = value_positions[var_a]

            if random.random() < false_a_ratio:
                pos_a = random.choice([p for p in range(1, self.N + 1) if p != true_pos])
            else:
                pos_a = true_pos
            clause_a = (var_a == pos_a)

            # 构造 B 子句（从不同属性和值中抽取）
            attr1, attr2 = random.sample(self.attributes, 2)
            val1 = random.choice(list(self.z3_vars[attr1].keys()))
            val2 = random.choice(list(self.z3_vars[attr2].keys()))
            var1 = self.z3_vars[attr1][val1]
            var2 = self.z3_vars[attr2][val2]

            b_type = random.choice(["adjacent", "greater", "notequal"])
            if b_type == "adjacent":
                clause_b = Abs(var1 - var2) == 1
            elif b_type == "greater":
                clause_b = var1 > var2
            else:  # "notequal"
                clause_b = var1 != var2

            if pos_a == true_pos and not is_clause_true_in_solution(clause_b):
                continue

            constraints.append(Implies(clause_a, clause_b))

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
            "IMPLICATE":self.original_imply_clues
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
            "IMPLICATE":0.98
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
    zebra_gen02 = Zebra_Gen02(3,3)
    print("===== original_solution =====")
    for i, h in enumerate(zebra_gen02.original_solution):
        print(f"House {i+1}: {h}")

    print("===== final Z3 Constraints =====")
    for i, c in enumerate(zebra_gen02.final_clues):
        print(f"{i+1}. {c}")

    print("===== check =====")
    zebra_gen02.verify_final_clues_solution_matches_original()




