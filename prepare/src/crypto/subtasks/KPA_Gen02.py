from KPA_Gen01 import KPA_Gen01
import random
import string
from typing import Tuple, Dict, List

'''
- task: Crypto-KPA
- subtask: Two layer encryption
- introduction: Encrypt twice and give two examples for analysis
'''

class KPA_Gen02(KPA_Gen01):
    def __init__(self, layers, parts):
        super().__init__(layers, parts)

    def apply_two_layer_encryption(self, text: str, methods: List[str], extra_params: Dict) -> str:
        for method in methods:
            if method == "caesar_shift_fixed":
                text = self.caesar_shift_fixed(text)
            elif method == "caesar_shift":
                text = self.caesar_shift(text, extra_params["shift"])
            elif method == "atbash_cipher":
                text = self.atbash_cipher(text)
            elif method == "random_substitution":
                text = self.random_substitution(text, extra_params["cipher_map"])
            elif method == "reverse_cipher":
                text = self.reverse_cipher(text)
            elif method == "rail_fence":
                text = self.rail_fence(text, extra_params["rails"])
            elif method == "columnar_transposition":
                text = self.columnar_transposition(text, extra_params["key"])
        return text

    def generate_puzzle(self) -> Tuple[Dict, Dict]:
        methods = random.sample(self.encryption_methods, 2)

        plaintexts_len = 20
        key_len = 0
        if "columnar_transposition" in methods:
            key_len = random.randint(3, 4)
            plaintexts_len = key_len * random.randint(3, 5)

        plaintexts = []
        if "random_substitution" in methods:
            pt = self.generate_plaintext(plaintexts_len)
            plaintexts.append(pt)
            unique_letters = list(set(pt))
            for _ in range(2):
                pt2 = ''.join(random.choices(unique_letters, k=len(pt)))
                plaintexts.append(pt2)
        else:
            for _ in range(3):
                plaintexts.append(self.generate_plaintext(plaintexts_len))

        extra_params = {}
        if "caesar_shift" in methods:
            extra_params["shift"] = random.randint(1, 6)
        if "random_substitution" in methods:
            letters = list(string.ascii_uppercase)
            shuffled = letters.copy()
            random.shuffle(shuffled)
            extra_params["cipher_map"] = dict(zip(letters, shuffled))
        if "rail_fence" in methods:
            extra_params["rails"] = random.randint(2, 3)
        if "columnar_transposition" in methods:
            extra_params["key"] = ''.join(random.sample(string.ascii_uppercase, key_len))

        ciphertexts = [self.apply_two_layer_encryption(pt, methods, extra_params) for pt in plaintexts]

        puzzle = ("Candidate methods:\n" + f"- {methods[0]}\n- {methods[1]}\n\n"+
                  f"Encryption sample:\n- {plaintexts[0]}->{ciphertexts[0]}\n- {plaintexts[1]}->{ciphertexts[1]}\n\n" + f"Cipher: {ciphertexts[2]}")
        solution = plaintexts[2]
        return puzzle, solution

if __name__ == "__main__":
    KPA_Gen02 = KPA_Gen02(2,1)
    puzzle, solution = KPA_Gen02.generate_puzzle()
    print(puzzle)
    print(solution)
