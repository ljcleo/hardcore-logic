from KPA_Gen01 import KPA_Gen01
import random
import string
from typing import Tuple, Dict, List

'''
- task: Crypto
- subtask: Unsolvable puzzle
- introduction: The two example encryption methods given are inconsistent
'''
class KPA_Uns(KPA_Gen01):
    def __init__(self,layers,parts):
        super().__init__(layers,parts)

    def _encrypt_with_method(self, plaintext: str, method: str) -> Tuple[str, Dict]:
        if method == "caesar_shift_fixed":
            return self.caesar_shift(plaintext, 3), {"type": "caesar_fixed", "shift": 3}
        elif method == "caesar_shift":
            shift = random.randint(1, 25)
            return self.caesar_shift(plaintext, shift), {"type": "caesar", "shift": shift}
        elif method == "atbash_cipher":
            return self.atbash_cipher(plaintext), {"type": "atbash"}
        elif method == "random_substitution":
            letters = list(string.ascii_uppercase)
            shuffled = letters.copy()
            random.shuffle(shuffled)
            cipher_map = dict(zip(letters, shuffled))
            return self.random_substitution(plaintext, cipher_map), {"type": "substitution", "mapping": cipher_map}
        elif method == "reverse_cipher":
            return self.reverse_cipher(plaintext), {"type": "reverse"}
        elif method == "rail_fence":
            rails = random.randint(2, 3)
            return self.rail_fence(plaintext, rails), {"type": "rail_fence", "rails": rails}
        else:
            raise ValueError(f"Unknown encryption method: {method}")

    def generate_unsolvable_puzzle(self) -> Tuple[Dict, Dict]:
        method1 = random.choice(self.encryption_methods)
        method2 = random.choice(self.encryption_methods)
        while method2 == method1:
            method2 = random.choice(self.encryption_methods)

        if method1 =="columnar_transposition":
            key = ''.join(random.sample(string.ascii_uppercase, random.randint(3, 4)))
            key_len = len(key)
            adjusted_length=key_len * random.randint(3, 5)
            plaintext1 = self.generate_plaintext(adjusted_length)
            ciphertext1 = self.columnar_transposition(plaintext1, key)
        else:
            plaintext1 = self.generate_plaintext()
            ciphertext1, details1 = self._encrypt_with_method(plaintext1, method1)

        if method2 =="columnar_transposition":
            key2 = ''.join(random.sample(string.ascii_uppercase, random.randint(3, 4)))
            key2_len = len(key2)
            adjusted_length2=key2_len * random.randint(3, 5)
            plaintext2 = self.generate_plaintext(adjusted_length2)
            ciphertext2 = self.columnar_transposition(plaintext2, key2)
        else:
            plaintext2 = self.generate_plaintext()
            ciphertext2, details2 = self._encrypt_with_method(plaintext2, method2)

        ciphertext3 = self.generate_plaintext()

        puzzle = f"Candidate methods:\n- {method1}\n- {method2}\n\n" + f"Encryption sample:\n- {plaintext1}->{ciphertext1}\n- {plaintext2}->{ciphertext2}\n\n"+ f"Cipher: {ciphertext3}"
        solution = None
        return puzzle, solution

if __name__ == "__main__":
    KPA_Uns = KPA_Uns(2,1)
    puzzle,solution = KPA_Uns.generate_unsolvable_puzzle()
    print(puzzle)
    print(solution)