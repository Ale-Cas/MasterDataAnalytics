"""
@author: Alessio Castrica
@date: 03/03/2022

https://christopherdare.com/code/aes
"""

from typing import List


class AESBlockCipher:
    def __init__(
        self,
        state: List[int],
        key: List[int],
    ) -> None:
        super().__init__()
        self.state = state
        self.key = key

    @property
    def num_rounds(self) -> int:
        return self.key + 6


if __name__ == "__main__":
    aes = AESBlockCipher(
        state=[
            [0x32, 0x43, 0xF6, 0xA8],
            [0x88, 0x5A, 0x30, 0x8D],
            [0x31, 0x31, 0x98, 0xA2],
            [0xE0, 0x37, 0x07, 0x34],
        ],
        key=[
            0x2B,
            0x7E,
            0x1C,
            0x15,
            0x18,
            0xAE,
            0xD2,
            0xA6,
            0xAE,
            0xF7,
            0x1B,
            0x88,
            0x09,
            0xC6,
            0x4F,
            0x3C,
            0x4B,
            0x7E,
            0x1C,
            0x15,
            0x18,
            0x3E,
            0xD2,
            0xA6,
            0xAE,
            0xF7,
            0x1B,
            0x68,
            0x09,
            0xCF,
            0x4F,
            0x3C,
        ],
    )
    print(aes.state)
