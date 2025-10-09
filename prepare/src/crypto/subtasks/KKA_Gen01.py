import random
import string
from typing import Tuple, Dict, List

'''
- task: Crypto-KKA
- subtask: More random plaintexts
- introduction: Each character in plaintexts is randomly generated without any rules
'''

class KKA_Gen01:
    def __init__(self,layers,parts):
        self.layers = layers
        self.parts = parts
        self.encryption_methods = [
            "caesar_shift_fixed",
            "reverse_cipher",
            "atbash_cipher",
            "caesar_shift",
            "rail_fence",
            "random_substitution",
            "columnar_transposition"
        ]

    def generate_plaintext(self, length: int = 20) -> str:
        return ''.join(random.choices(string.ascii_uppercase, k=length))

    def caesar_shift_fixed(self, text: str, shift: int = 3) -> str:
        return self.caesar_shift(text, shift)

    def caesar_shift(self, text: str, shift: int) -> str:
        result = []
        for char in text:
            shifted = (ord(char) - ord('A') + shift) % 26
            result.append(chr(shifted + ord('A')))
        return ''.join(result)

    def atbash_cipher(self, text: str) -> str:
        return ''.join([chr(ord('Z') - (ord(c) - ord('A'))) for c in text])

    def reverse_cipher(self, text: str) -> str:
        return text[::-1]

    def rail_fence(self, text: str, rails: int) -> str:
        fence = [[] for _ in range(rails)]
        rail, direction = 0, 1
        for char in text:
            fence[rail].append(char)
            rail += direction
            if rail == rails - 1 or rail == 0:
                direction *= -1
        return ''.join([''.join(row) for row in fence])

    def random_substitution(self, text: str, cipher_map: Dict[str, str]) -> str:
        return ''.join([cipher_map[c] for c in text])

    def columnar_transposition(self, text: str, key: str) -> str:
        key_len = len(key)
        columns = ['' for _ in range(key_len)]
        for idx, char in enumerate(text):
            col = idx % key_len
            columns[col] += char
        key_order = sorted(range(len(key)), key=lambda k: key[k])
        return ''.join(columns[i] for i in key_order)

    def generate_puzzle(self) -> Tuple[Dict, Dict]:
        method = random.choice(self.encryption_methods)
        plaintext = self.generate_plaintext()

        if method == "caesar_shift_fixed":
            ciphertext = self.caesar_shift_fixed(plaintext)
            cipher_description = "Caesar cipher with fixed shift of 3"

        elif method == "caesar_shift":
            shift = random.randint(1, 25)
            ciphertext = self.caesar_shift(plaintext, shift)
            cipher_description = f"Caesar cipher with shift of {shift}"

        elif method == "atbash_cipher":
            ciphertext = self.atbash_cipher(plaintext)
            cipher_description = "Atbash cipher (each letter replaced by its opposite in the alphabet)"

        elif method == "reverse_cipher":
            ciphertext = self.reverse_cipher(plaintext)
            cipher_description = "Reverse cipher (the text is reversed)"

        elif method == "rail_fence":
            rails = random.randint(2, 3)
            ciphertext = self.rail_fence(plaintext, rails)
            cipher_description = f"Rail fence cipher with {rails} rails"

        elif method == "random_substitution":
            letters = list(string.ascii_uppercase)
            shuffled = letters[:]
            random.shuffle(shuffled)
            cipher_map = dict(zip(letters, shuffled))
            mapping_str = ", ".join([f"{k}→{v}" for k, v in sorted(cipher_map.items())])
            ciphertext = self.random_substitution(plaintext, cipher_map)
            cipher_description = f"Monoalphabetic substitution cipher with known mapping:\n{mapping_str}"

        elif method == "columnar_transposition":
            key_len = random.randint(3, 4)
            key = ''.join(random.sample(string.ascii_uppercase, key_len))
            multiplier = random.randint(2, 5)
            plaintext = self.generate_plaintext(length=key_len * multiplier)
            ciphertext = self.columnar_transposition(plaintext, key)
            cipher_description = f"Columnar transposition cipher with key = '{key}'"

        puzzle = "Candidate methods:\n" +f'- {cipher_description}\n' + "\n" + "Encryption sample: (not available)\n\n" + f'Cipher: {ciphertext}'
        solution = plaintext
        return puzzle, solution

if __name__ == "__main__":
    puzzle, solution = KKA_Gen01(1,1).generate_puzzle()
    print(puzzle)
    print(solution)