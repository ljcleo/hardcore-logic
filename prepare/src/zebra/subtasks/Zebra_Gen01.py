from z3 import Solver, Int, Distinct, And, Or, Not, sat, Abs, unsat,Bool,Implies
import random
from copy import deepcopy

'''
- task : zebra
- subtask : create harder clues
- introduction : In this section, we increase the difficulty of the questions by
reducing the information of the constraints and increasing the conflicts between the constraints.
'''

class Zebra_Gen01:
    def __init__(self, N=3, M=3, num=-1, num_fakes=-1, seed=None):
        self.N = N      #The number of houses
        self.M = M      #The number of attribution
        self.num = num
        self.num_fakes = num_fakes
        self.seed = seed if seed is not None else random.randint(1, 10000)
        random.seed(self.seed)
        #Attributions
        self.A_all = ["Name", "Color", "Nationality", "Animal", "Drink", "Cigar", "Food", "Flower",
                      "PhoneModel", "Children", "Smoothie", "Birthday", "Occupation", "Height", "CarModel",
                      "FavoriteSport", "MusicGenre", "BookGenre", "HairColor", "Mother", "HouseStyle",
                      "Education", "Hobby", "Vacation", "Pet"]
        #All possible values of each attribute
        self.attribute_values = {
            "Name": ["Allen", "Mike", "Sarah", "Linda", "John", "Emily"],
            "Color": ["Red", "Blue", "Green", "Yellow", "White", "Black"],
            "Nationality": ["American", "British", "Chinese", "German", "French", "Japanese"],
            "Animal": ["Dog", "Cat", "Rabbit", "Parrot", "Hamster", "Turtle"],
            "Drink": ["Water", "Tea", "Coffee", "Juice", "Milk", "Soda"],
            "Cigar": ["Pall Mall", "Prince", "Dunhill", "Marlboro", "Winston", "Camel"],
            "Food": ["Pizza", "Burger", "Sushi", "Pasta", "Salad", "Steak"],
            "Flower": ["Rose", "Tulip", "Lily", "Daisy", "Orchid", "Sunflower"],
            "PhoneModel": ["iPhone", "Galaxy", "Pixel", "OnePlus", "Huawei", "Xiaomi"],
            "Children": ["No kids", "1 child", "2 children", "3 children", "Twins", "Adopted child"],
            "Smoothie": ["Strawberry", "Mango", "Banana", "Blueberry", "Avocado", "Pineapple"],
            "Birthday": ["Jan 1", "Feb 14", "Mar 8", "Apr 22", "May 5", "Jun 30"],
            "Occupation": ["Engineer", "Teacher", "Artist", "Doctor", "Chef", "Lawyer"],
            "Height": ["Short", "Average", "Tall", "Very tall", "Petite", "Medium"],
            "CarModel": ["Tesla", "BMW", "Toyota", "Ford", "Audi", "Honda"],
            "FavoriteSport": ["Soccer", "Basketball", "Tennis", "Baseball", "Swimming", "Running"],
            "MusicGenre": ["Rock", "Pop", "Jazz", "Classical", "Hip-hop", "Electronic"],
            "BookGenre": ["Mystery", "Sci-Fi", "Romance", "Fantasy", "Biography", "History"],
            "HairColor": ["Blonde", "Brown", "Black", "Red", "Gray", "Blue"],
            "Mother": ["Anna", "Beth", "Carla", "Diana", "Eva", "Fiona"],
            "HouseStyle": ["Cottage", "Villa", "Apartment", "Bungalow", "Mansion", "Cabin"],
            "Education": ["High School", "Bachelor", "Master", "PhD", "Diploma", "None"],
            "Hobby": ["Gardening", "Cooking", "Painting", "Photography", "Dancing", "Hiking"],
            "Vacation": ["Paris", "Tokyo", "New York", "Rome", "Sydney", "Cairo"],
            "Pet": ["Dog", "Cat", "Fish", "Bird", "Hamster", "Snake"]
        }

        self.attributes = self.generate_attributes()
        self.original_solution = self.generate_original_solution()     #generate original solution
        self.init_z3_vars()

        #generate various categories of constraints
        self.original_foundat_clues = self.generate_foundat_clues()
        self.original_samehouse_clues = self.generate_samehouse_clues()
        self.original_notat_clues=self.generate_notat_clues()
        self.original_directadj_clues=self.generate_directadj_clues()
        self.original_left_rightof_clues=self.generate_left_rightof_clues()
        self.original_between_clues=self.generate_between_clues()
        self.original_sidebyside_clues=self.generate_sidebyside_clues()
        self.global_constraints=self.generate_global_constraints()

        #Minimize clues while ensure the solution
        self.final_clues=self.simplify_clues()

    def generate_attributes(self):
        others = [attr for attr in self.A_all if attr != "Name"]
        selected = random.sample(others, self.M - 1)
        return ["Name"] + selected

    def generate_original_solution(self):
        solution = []
        shuffled_values = {}

        for attr in self.attributes:
            pool = self.attribute_values[attr]
            if len(pool) < self.N:
                raise ValueError(f"Attribute '{attr}' has insufficient values.")
            shuffled_values[attr] = random.sample(pool, self.N)

        for i in range(self.N):
            house = {attr: shuffled_values[attr][i] for attr in self.attributes}
            solution.append(house)

        return solution

    def init_z3_vars(self):
        self.z3_vars = {}
        for attr in self.attributes:
            self.z3_vars[attr] = {}
            used_values = {row[attr] for row in self.original_solution}
            for value in used_values:
                self.z3_vars[attr][value] = Int(f"{attr}_{value}")

    def generate_foundat_clues(self):
        constraints = []

        for house_idx in range(self.N):
            for attr in self.attributes:
                true_value = self.original_solution[house_idx][attr]
                var_true = self.z3_vars[attr][true_value]

                used_values = {row[attr] for row in self.original_solution}
                for other_value in used_values:
                    if other_value != true_value:
                        var_false = self.z3_vars[attr][other_value]
                        constraints.append(var_false != house_idx + 1)

        return constraints

    def generate_samehouse_clues(self):
        constraints = []
        for house in self.original_solution:
            for i in range(len(self.attributes)):
                for j in range(i + 1, len(self.attributes)):
                    attr1 = self.attributes[i]
                    attr2 = self.attributes[j]
                    value1 = house[attr1]
                    value2 = house[attr2]

                    var1 = self.z3_vars[attr1][value1]
                    var2 = self.z3_vars[attr2][value2]
                    constraints.append(var1 == var2)
        return constraints

    def generate_notat_clues(self):
        constraints = []
        for i in range(self.N):
            for j in range(self.N):
                if i == j:
                    continue
                house_i = self.original_solution[i]
                house_j = self.original_solution[j]

                for attr1 in self.attributes:
                    for attr2 in self.attributes:
                        if attr1 == attr2:
                            continue

                        value1 = house_i[attr1]
                        value2 = house_j[attr2]

                        var1 = self.z3_vars[attr1][value1]
                        var2 = self.z3_vars[attr2][value2]
                        constraints.append(var1 != var2)
        return constraints

    def generate_directadj_clues(self):
        constraints = []

        for i in range(self.N - 1):
            house_left = self.original_solution[i]
            house_right = self.original_solution[i + 1]

            for attr1 in self.attributes:
                val1 = house_left[attr1]
                var1 = self.z3_vars[attr1][val1]

                for attr2 in self.attributes:
                    val2 = house_right[attr2]
                    var2 = self.z3_vars[attr2][val2]

                    constraints.append(var1 + 1 == var2)

                    constraints.append(var2 - 1 == var1)

        return constraints

    def generate_left_rightof_clues(self):
        constraints = []

        for i in range(self.N):
            for j in range(self.N):
                if i == j:
                    continue

                house_i = self.original_solution[i]
                house_j = self.original_solution[j]

                for attr1 in self.attributes:
                    for attr2 in self.attributes:
                        if attr1 == attr2:
                            continue

                        value1 = house_i[attr1]
                        value2 = house_j[attr2]

                        var1 = self.z3_vars[attr1][value1]
                        var2 = self.z3_vars[attr2][value2]

                        if i < j:
                            constraints.append(var1 < var2)
                        else:
                            constraints.append(var1 > var2)

        return constraints

    def generate_between_clues(self):
        constraints = []

        for i in range(self.N):
            for j in range(self.N):
                if i == j:
                    continue

                dist = abs(i - j)
                if dist not in [2, 3]:
                    continue

                house_i = self.original_solution[i]
                house_j = self.original_solution[j]

                for attr1 in self.attributes:
                    for attr2 in self.attributes:
                        if attr1 == attr2:
                            continue

                        value1 = house_i[attr1]
                        value2 = house_j[attr2]

                        var1 = self.z3_vars[attr1][value1]
                        var2 = self.z3_vars[attr2][value2]

                        if dist == 2:
                            clue_type = "ONEBETWEEN"
                            constraints.append(Abs(var1 - var2) == 2)
                        elif dist == 3:
                            clue_type = "TWOBETWEEN"
                            constraints.append(Abs(var1 - var2) == 3)

        return constraints

    def generate_sidebyside_clues(self):
        constraints = []
        used_values_by_attr = {
            attr: {row[attr] for row in self.original_solution}
            for attr in self.attributes
        }

        for i, attr1 in enumerate(self.attributes):
            for attr2 in self.attributes[i:]:
                for val1 in used_values_by_attr[attr1]:
                    for val2 in used_values_by_attr[attr2]:
                        var1 = self.z3_vars[attr1][val1]
                        var2 = self.z3_vars[attr2][val2]
                        pos1 = next(i for i, h in enumerate(self.original_solution) if h[attr1] == val1)
                        pos2 = next(i for i, h in enumerate(self.original_solution) if h[attr2] == val2)
                        if abs(pos1 - pos2) == 1:
                            constraints.append(Abs(var1 - var2) == 1)

        return constraints

    def generate_global_constraints(self):
        constraints = []
        for attr in self.attributes:
            vars_list = []
            used_values = {row[attr] for row in self.original_solution}  # 只使用当前谜题中实际出现的值
            for value in used_values:
                var = self.z3_vars[attr][value]
                constraints.append(And(var >= 1, var <= self.N))
                vars_list.append(var)
            constraints.append(Distinct(vars_list))
        return constraints

    def simplify_clues(self):

        # 1. Integrate all clues
        all_clues = []
        clues_by_type = {
            "FOUNDAT": self.original_foundat_clues,
            "SAMEHOUSE": self.original_samehouse_clues,
            "NOTAT": self.original_notat_clues,
            "DIRECTADJ": self.original_directadj_clues,
            "LEFTRIGHTOF": self.original_left_rightof_clues,
            "BETWEEN": self.original_between_clues,
            "SIDEBYSIDE":self.original_sidebyside_clues,
        }

        for clue_type, clues in clues_by_type.items():
            for clue in clues:
                all_clues.append({
                    "type": clue_type,
                    "expr": clue
                })

        # 2. Set the deletion preference weight for lead types
        type_weights = {
            "FOUNDAT": 0.99,
            "SAMEHOUSE": 1,
            "NOTAT": 1,
            "DIRECTADJ": 0.995,
            "LEFTRIGHTOF": 0.995,
            "BETWEEN": 0.99,
            "SIDEBYSIDE":0.99,
        }

        # 3. Initialize the clue set
        current_clues = deepcopy(all_clues)

        # 4. Used for unique solution determination
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

        # 5. Main loop: keep trying to delete clues
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

        # 6. Returns the simplified clue set
        simplified = [c["expr"] for c in current_clues]
        return simplified


    def verify_final_clues_solution_matches_original(self):
        """
        Check whether the simplified clue set can uniquely solve original_solution.
        """
        solver = Solver()
        solver.add(self.global_constraints)
        solver.add(self.final_clues)

        if solver.check() != sat:
            print("❌ Final clues cannot produce a valid solution.")
            return False

        model = solver.model()

        for i in range(self.N):
            for attr in self.attributes:
                val = self.original_solution[i][attr]
                var = self.z3_vars[attr][val]
                assigned = model.eval(var).as_long()
                if assigned != i + 1:
                    print(f"Mismatch: {attr}-{val} is in house {assigned}, expected house {i + 1}")
                    return False

        must_match = [self.z3_vars[attr][self.original_solution[i][attr]] == i + 1
                      for i in range(self.N) for attr in self.attributes]

        solver.push()
        solver.add(Not(And(must_match)))
        if solver.check() == sat:
            print("Final clues have multiple valid solutions.")
            solver.pop()
            return False
        solver.pop()

        print("Final clues uniquely and correctly determine the original solution.")
        return True

if __name__ == "__main__":
    zebra_gen01 = Zebra_Gen01(3,3)
    print("===== original_solution =====")
    for i, h in enumerate(zebra_gen01.original_solution):
        print(f"House {i+1}: {h}")

    print("===== final Z3 Constraints =====")
    for i, c in enumerate(zebra_gen01.final_clues):
        print(f"{i+1}. {c}")

    print("===== check =====")
    zebra_gen01.verify_final_clues_solution_matches_original()
