from KKA_Gen01 import KKA_Gen01
import random
import string
from typing import Tuple, Dict, List

'''
- task: Crypto-KKA
- subtask: Multi-layers
- introduction: Perform multiple encryption
'''
class KKA_Gen03(KKA_Gen01):
    def __init__(self,layers,parts):
        super().__init__(layers,parts)

    def apply_encryption_chain(self, text: str, methods: List[str]) -> Tuple[str, List[Dict]]:
        method_infos = []
        for method in methods:
            if method == "caesar_shift_fixed":
                text = self.caesar_shift_fixed(text)
                method_infos.append({"type": "caesar_fixed", "shift": 3,'description': "Caesar cipher with fixed shift of 3"})

            elif method == "caesar_shift":
                shift = random.randint(1, 25)
                text = self.caesar_shift(text, shift)
                method_infos.append({"type": "caesar", "shift": shift,'description':f"Caesar cipher with shift of {shift}"})

            elif method == "atbash_cipher":
                text = self.atbash_cipher(text)
                method_infos.append({"type": "atbash", 'description':"Atbash cipher (each letter replaced by its opposite in the alphabet)"})

            elif method == "reverse_cipher":
                text = self.reverse_cipher(text)
                method_infos.append({"type": "reverse",'description':"Reverse cipher (the text is reversed)"})

            elif method == "rail_fence":
                rails = random.randint(2, 3)
                text = self.rail_fence(text, rails)
                method_infos.append({"type": "rail_fence", "rails": rails,'description':f"Rail fence cipher with {rails} rails"})

            elif method == "random_substitution":
                letters = list(string.ascii_uppercase)
                shuffled = letters[:]
                random.shuffle(shuffled)
                cipher_map = dict(zip(letters, shuffled))
                mapping_str = ", ".join([f"{k}→{v}" for k, v in sorted(cipher_map.items())])
                text = self.random_substitution(text, cipher_map)
                method_infos.append({"type": "substitution", "mapping": cipher_map,'description':f"Monoalphabetic substitution cipher with known mapping:\n{mapping_str}"})

            elif method == "columnar_transposition":
                key_len = random.randint(3, 4)
                key = ''.join(random.sample(string.ascii_uppercase, key_len))
                text = self.columnar_transposition(text, key)
                method_infos.append({"type": "columnar", "key": key,'description':f"Columnar transposition cipher with key = '{key}'"})

        return text, method_infos

    def generate_puzzle(self) -> Tuple[Dict, Dict]:
        chosen_methods = random.sample(self.encryption_methods, self.layers)

        # 为 columnar_transposition 兼容设置明文长度
        base_length = 12 * random.randint(1, 3)
        plaintext = self.generate_plaintext(length=base_length)

        ciphertext, method_chain = self.apply_encryption_chain(plaintext, chosen_methods)

        puzzle = "Candidate methods:\n"
        for m in method_chain:
            puzzle += f"- {m['description']}\n"

        puzzle += "\nEncryption sample: (not available)\n\n"
        puzzle += f"Cipher: {ciphertext}"
        solution = plaintext
        return  puzzle, solution

if __name__ == "__main__":
    KKA_Gen03 = KKA_Gen03(3,1)
    puzzle, solution = KKA_Gen03.generate_puzzle()
    print(puzzle)
    print(solution)