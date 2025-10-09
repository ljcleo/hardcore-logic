from KKA_Gen01 import KKA_Gen01
import random
import string
from typing import Tuple, Dict, List

'''
- task: Crypto-KKA
- subtask: Multi-parts
- introduction: Perform multiple segments encryption
'''
class KKA_Gen04(KKA_Gen01):
    def __init__(self, layers,parts):
        super().__init__(layers,parts)

    def encrypt(self, text: str, method: str) -> Tuple[str, Dict]:
        if method == "caesar_shift_fixed":
            return self.caesar_shift_fixed(text), "Caesar cipher with fixed shift of 3"

        elif method == "caesar_shift":
            shift = random.randint(1, 25)
            return self.caesar_shift(text, shift), f"Caesar cipher with shift of {shift}"

        elif method == "atbash_cipher":
            return self.atbash_cipher(text), "Atbash cipher (each letter replaced by its opposite in the alphabet)"

        elif method == "reverse_cipher":
            return self.reverse_cipher(text), "Reverse cipher (the text is reversed)"

        elif method == "rail_fence":
            rails = random.randint(2, 3)
            return self.rail_fence(text, rails), f"Rail fence cipher with {rails} rails"

        elif method == "random_substitution":
            letters = list(string.ascii_uppercase)
            shuffled = letters[:]
            random.shuffle(shuffled)
            cipher_map = dict(zip(letters, shuffled))
            mapping_str = ", ".join([f"{k}→{v}" for k, v in sorted(cipher_map.items())])
            return self.random_substitution(text, cipher_map),  f"Monoalphabetic substitution cipher with known mapping:\n{mapping_str}"

        elif method == "columnar_transposition":
            key_len = random.randint(3, 4)
            key = ''.join(random.sample(string.ascii_uppercase, key_len))
            length = len(text)
            length = key_len * (length // key_len)
            text = text[:length]
            return self.columnar_transposition(text, key), f"Columnar transposition cipher with key = '{key}'"

        else:
            raise ValueError(f"Unsupported encryption method: {method}")

    def generate_puzzle(self) -> Tuple[Dict, Dict]:
        all_methods = self.encryption_methods

        methods = random.sample(all_methods, self.parts)

        def get_valid_length_for_method(method: str) -> Tuple[int, int]:
            if method == "columnar_transposition":
                key_len = random.randint(3, 4)
                base = random.randint(3, 5)
                length = key_len * base
                return length, key_len
            elif method == "rail_fence":
                return random.randint(6, 10), None
            else:
                return random.randint(6, 10), None

        def encrypt_with_optional_key(text: str, method: str, key_len=None) -> Tuple[str, Dict]:
            if method == "columnar_transposition":
                key = ''.join(random.sample(string.ascii_uppercase, key_len))
                ciphertext = self.columnar_transposition(text, key)
                return ciphertext,f"Columnar transposition cipher with key = '{key}'"
            else:
                return self.encrypt(text, method)

        plaintext_parts = []
        cipher_parts = []
        methods_info = []

        for method in methods:
            length, key_len = get_valid_length_for_method(method)
            part = self.generate_plaintext(length)
            cipher, info = encrypt_with_optional_key(part, method, key_len)

            plaintext_parts.append(part)
            cipher_parts.append(cipher)
            methods_info.append(info)

        full_plaintext = "".join(plaintext_parts)
        full_cipher = "|".join(cipher_parts)

        puzzle = "Candidate methods:\n- " + " + ".join(info for info in methods_info) + "\n\nEncryption sample: (not available)\n\n"+ f'Cipher: {full_cipher}'

        answer = {
            "plaintext": full_plaintext,
            "parts": plaintext_parts
        }

        return puzzle, answer

if __name__ == "__main__":
    KKA_gen04 = KKA_Gen04(1,2)
    puzzle, answer = KKA_gen04.generate_puzzle()
    print(puzzle)
    print(answer)
