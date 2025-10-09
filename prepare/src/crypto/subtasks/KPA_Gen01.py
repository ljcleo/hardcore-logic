import random
import string
from typing import Dict, Tuple

'''
- task: Crypto-KPA
- subtask: More random plaintexts
- introduction: Each character in plaintexts is randomly generated without any rules
'''

class KPA_Gen01:
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
        return ''.join(chr((ord(c)-ord('A')+shift)%26+ord('A')) for c in text)

    def atbash_cipher(self, text: str) -> str:
        return ''.join(chr(ord('Z')-(ord(c)-ord('A'))) for c in text)

    def random_substitution(self, text: str, cipher_map: Dict[str,str]) -> str:
        return ''.join(cipher_map[c] for c in text)

    def reverse_cipher(self, text: str) -> str:
        return text[::-1]

    def rail_fence(self, text: str, rails: int) -> str:
        fence = [[] for _ in range(rails)]
        rail, direction = 0, 1
        for char in text:
            fence[rail].append(char)
            rail += direction
            if rail == rails-1 or rail==0:
                direction *= -1
        return ''.join(''.join(row) for row in fence)

    def columnar_transposition(self, text: str, key: str) -> str:
        key_len = len(key)
        padded_text = text + 'X' * (key_len - len(text)%key_len) if len(text)%key_len != 0 else text
        columns = [padded_text[i::key_len] for i in range(key_len)]
        sorted_columns = [col for _, col in sorted(zip(key, columns))]
        return ''.join(sorted_columns)

    def generate_puzzle(self) -> Tuple[Dict, Dict]:
        method = random.choice(self.encryption_methods)

        plaintext1 = self.generate_plaintext()

        if method == "random_substitution":
            unique_chars = list(set(plaintext1))
            plaintext2 = ''.join(random.choices(unique_chars, k=len(plaintext1)))
        else:
            plaintext2 = self.generate_plaintext()

        if method == "caesar_shift_fixed":
            ciphertext1 = self.caesar_shift_fixed(plaintext1)
            ciphertext2 = self.caesar_shift_fixed(plaintext2)
        elif method == "caesar_shift":
            shift = random.randint(1,25)
            ciphertext1 = self.caesar_shift(plaintext1, shift)
            ciphertext2 = self.caesar_shift(plaintext2, shift)
        elif method == "atbash_cipher":
            ciphertext1 = self.atbash_cipher(plaintext1)
            ciphertext2 = self.atbash_cipher(plaintext2)
        elif method == "random_substitution":
            letters = list(string.ascii_uppercase)
            shuffled = letters.copy()
            random.shuffle(shuffled)
            cipher_map = dict(zip(letters, shuffled))
            ciphertext1 = self.random_substitution(plaintext1, cipher_map)
            ciphertext2 = self.random_substitution(plaintext2, cipher_map)
        elif method == "reverse_cipher":
            ciphertext1 = self.reverse_cipher(plaintext1)
            ciphertext2 = self.reverse_cipher(plaintext2)
        elif method == "rail_fence":
            rails = random.randint(2,3)
            ciphertext1 = self.rail_fence(plaintext1, rails)
            ciphertext2 = self.rail_fence(plaintext2, rails)
        elif method == "columnar_transposition":
            key = ''.join(random.sample(string.ascii_uppercase, random.randint(3,4)))
            key_len = len(key)
            valid_length = key_len * random.randint(3,5)
            plaintext1 = self.generate_plaintext(valid_length)
            plaintext2 = self.generate_plaintext(valid_length)
            ciphertext1 = self.columnar_transposition(plaintext1, key)
            ciphertext2 = self.columnar_transposition(plaintext2, key)

        puzzle = 'Candidate methods: (not available)\n' + "\n" + f'Encryption sample:\n'+f'- {plaintext1} -> {ciphertext1}\n\n' + f'Cipher: {ciphertext2}'
        solution = plaintext2
        return puzzle, solution

if __name__ == "__main__":
    kpa = KPA_Gen01(1,1)
    puzzle, solution = kpa.generate_puzzle()
    print(puzzle)
    print(solution)