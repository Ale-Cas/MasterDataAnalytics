"""
@author: Alessio Castrica
@date: 28/02/2022
"""
from abc import ABC, abstractmethod
from typing import Dict, Tuple

import numpy as np


def random_bytes_matrix(
    num_rows: int = 4,
    num_cols: int = 4,
) -> np.matrix:
    """
    This function creates a numpy matrix of random integers in the range 0-255.

    Parameters
    ----------
    num_rows: int
        default = 4
    num_cols: int
        default = 4
    """
    return np.matrix(np.random.randint(256, size=(num_rows, num_cols)))


class BlockCipher(ABC):
    @abstractmethod
    def add_key(self) -> None:
        pass

    @abstractmethod
    def s_box(self) -> None:
        pass

    @abstractmethod
    def shift_row(self) -> None:
        pass

    @abstractmethod
    def mix_columns(self) -> None:
        pass

    @abstractmethod
    def key_schedule(self) -> None:
        pass

    @abstractmethod
    def encrypt(self) -> None:
        pass

    @abstractmethod
    def decrypt(self) -> None:
        pass


class AES(BlockCipher):
    def __init__(
        self,
        state: np.matrix = random_bytes_matrix(),
        key: np.matrix = random_bytes_matrix(num_rows=4, num_cols=8),
    ) -> None:
        super().__init__()
        self.state = state
        self.key = key

    @property
    def key_shape(self) -> Tuple[int, int]:
        return self.key.shape

    @property
    def key_size(self) -> int:
        r, c = self.key_shape
        return r * c * 8

    @property
    def num_cols_key(self) -> int:
        _, _num_cols_key = self.key_shape
        return _num_cols_key

    @property
    def num_rounds(self) -> int:
        return self.num_cols_key + 6

    def key_schedule(self, print_info: bool = True) -> Dict[int, np.matrix]:
        self.round_key = {}
        _key = self.key.transpose()
        if print_info:
            print(f"Initial key state:\n{_key}")
        for _ in range(self.num_rounds):
            update = np.bitwise_xor(_key[0], _key[-1])
            _key = np.insert(_key, 0, update, axis=0)
            _key = _key = np.delete(_key, -1, 0)
            if print_info:
                print(f"Key state after update n°{_+1}:\n{_key}")
            self.round_key[_] = _key
        return self.round_key

    def f_out(self, n_rows: int = 4, round: int = 0) -> np.matrix:
        return np.transpose(self.round_key[0][:-n_rows])

    def add_key(self, round: int = 0) -> np.matrix:
        return np.bitwise_xor(
            self.state,
            self.f_out(n_rows=self.num_cols_key - len(self.state), round=round),
        )

    def s_box(self) -> None:
        pass

    def shift_row(self) -> None:
        pass

    def mix_columns(self) -> None:
        pass

    def encrypt(self) -> None:
        pass

    def decrypt(self) -> None:
        pass

    @property
    def multiplication_lookup_table(self):
        pass

    @property
    def substitution_lookup_table(self):
        pass


if __name__ == "__main__":
    aes = AES()
    print(f"AES key size: {aes.key_size}")
    print("Key Schedule:\n")
    aes.key_schedule()
    print("Add Key:\n")
    print(aes.add_key())
    # print(f"Final key schedule:\n{aes.key_schedule()}")
