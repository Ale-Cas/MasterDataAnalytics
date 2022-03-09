"""
@author: Alessio Castrica
@date: 09/03/2022

Implementazione dell'algoritmo RSA.
"""
import math
import string
import random
from typing import List, Set, Tuple


def is_prime(n: int) -> bool:
    """
    Check wether a certain number is prime or not.

    Parameters
    ----------
    n: int
        The number to check.

    Returns
    -------
    bool
        True if the number is prime, False if it isn't.
    """
    if n <= 1:
        return False
    if n == 2:
        return True
    if n > 2 and n % 2 == 0:
        return False

    max_div = math.floor(math.sqrt(n))
    for i in range(3, 1 + max_div, 2):
        if n % i == 0:
            return False
    return True


def prime_nums_in_interval(upper_value: int, lower_value: int = 1) -> Set[int]:
    """
    Check wether a certain number is prime or not.

    Parameters
    ----------
    upper_value: int
        The maximum number to check (inclusive).
    lower_value: int
        The minimum number to check (inclusive).

    Returns
    -------
    primes: Set[int]
        A set of prime numbers.
    """
    primes = set()
    for n in range(lower_value, upper_value + 1):
        if is_prime(n):
            primes.add(n)
    return primes


def gcd(a, b):
    """
    Performs the Euclidean algorithm and
    returns the greatest common divisor of a and b.
    """
    if b == 0:
        return a
    else:
        return gcd(b, a % b)


def extended_gcd(a, b):
    if a == 0:
        return b, 0, 1
    else:
        g, y, x = extended_gcd(b % a, a)
        return g, x - (b // a) * y, y


def modular_inverse(a, m):
    g, x, _ = extended_gcd(a, m)
    if g != 1:
        raise Exception("modular inverse does not exist")
    else:
        return x % m


def text_to_digits(text: str) -> List[int]:
    pool = string.ascii_letters + string.punctuation + " "
    digits = []
    for i in text:
        digits.append(pool.index(i))
    return digits


def digits_to_text(digits: List[int]) -> str:
    pool = string.ascii_letters + string.punctuation + " "
    text = ""
    for i in digits:
        text += pool[i]
    return text


class RSA:
    def __init__(
        self,
        p: int = max(prime_nums_in_interval(random.randint(1e1, 1e2))),
        q: int = max(prime_nums_in_interval(random.randint(1e1, 1e2))),
    ) -> None:
        if isinstance(p, int) and isinstance(p, int):
            if is_prime(p):
                self.p = p
            else:
                raise ValueError("p must be a prime number")
            if is_prime(q):
                self.q = q
            else:
                raise ValueError("q must be a prime number")
        else:
            raise ValueError("p and q must be 2 integers")
        self._e = None
        self._d = None

    @property
    def n(self) -> int:
        """Public."""
        return self.p * self.q

    @property
    def phi_n(self) -> int:
        return (self.p - 1) * (self.q - 1)

    @property
    def e(self) -> int:
        if self._e is None:
            while True:
                e = random.randrange(2, self.phi_n)
                if gcd(e, self.phi_n) == 1:
                    self._e = e
                    break
        return self._e

    @property
    def d(self) -> int:
        if self._d is None:
            self._d = modular_inverse(self.e, self.phi_n)
        return self._d

    @property
    def public_key(self) -> Tuple[int, int]:
        return (self.e, self.n)

    @property
    def private_key(self) -> Tuple[int, int]:
        return (self.d, self.n)

    def encrypt(self, plaintext: str) -> List[int]:
        """
        Encrypts a message.

        Parameters
        ----------
        plaintext: str
        """
        m = text_to_digits(plaintext)
        return [(i ** self.public_key[0]) % self.public_key[1] for i in m]

    def decrypt(self, ciphertext: List[int]) -> str:
        """
        Decrypts an encrypted message.

        Parameters
        ----------
        ciphertext:  List[int]
        """
        c = [((i ** self.private_key[0]) % self.private_key[1]) for i in ciphertext]
        return digits_to_text(c)


if __name__ == "__main__":
    rsa = RSA()
    message = ""
    while len(message) == 0:
        message = str(input("Please enter a message: "))
    print(f"Message: {message}")
    print(f"Public Key: {rsa.public_key}")
    print(f"Private Key: {rsa.private_key}")
    encrypted_message = rsa.encrypt(message)
    print(f"Encrypted Message: {encrypted_message}")
    decrypted_message = rsa.decrypt(encrypted_message)
    print(f"Decrypted Message: {decrypted_message}")
