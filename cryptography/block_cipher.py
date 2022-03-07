"""
@author: Alessio Castrica
@date: 03/03/2022
"""
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


class AESBlockCipher:
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

    def key_schedule(self, print_info: bool = False) -> Dict[int, np.matrix]:
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

    def s_box(self) -> np.array:
        return np.array(
            [
                self.substitution_lookup_table.reshape((256))[b]
                for b in np.nditer(self.state)
            ],
        ).reshape(self.state.shape)

    def shift_rows(self) -> None:
        for r in range(len(self.state)):
            self.state[r] = np.roll(np.array(self.state[r]), r)
        return self.state

    def mix_columns(self) -> None:
        def xtime(x):
            return (x << 1) ^ (((x >> 7) & 0x01) * 0x1B)

        def gmult(x, y):
            result = (y & 0x01) * x
            result ^= (y >> 1 & 0x01) * xtime(x)
            result ^= (y >> 2 & 0x01) * xtime(xtime(x))
            result ^= (y >> 3 & 0x01) * xtime(xtime(xtime(x)))
            result ^= (y >> 4 & 0x01) * xtime(xtime(xtime(xtime(x))))
            return result & 0xFF

        _state = self.state
        for row in range(4):
            # temporary variables
            a = _state[row, 0]
            b = _state[row, 1]
            c = _state[row, 2]
            d = _state[row, 3]

            _state[row, 0] = gmult(a, 2) ^ gmult(b, 3) ^ c ^ d
            _state[row, 1] = a ^ gmult(b, 2) ^ gmult(c, 3) ^ d
            _state[row, 2] = a ^ b ^ gmult(c, 2) ^ gmult(d, 3)
            _state[row, 3] = gmult(a, 3) ^ b ^ c ^ gmult(d, 2)
        self.state = _state
        return self.state

    @property
    def substitution_lookup_table(self) -> np.ndarray:
        return np.genfromtxt(
            "cryptography\AESSubstitutionLookUpTable.csv",
            delimiter=",",
            dtype=np.int64,
        )


if __name__ == "__main__":
    aes = AESBlockCipher()
    print("Key Schedule:")
    aes.key_schedule(print_info=True)
    print("Add Key:")
    print(aes.add_key())
    print("S-box:")
    print(aes.s_box())
    print("Shift Rows:")
    print(aes.shift_rows())
    print("Mix Columns:")
    print(aes.mix_columns())
