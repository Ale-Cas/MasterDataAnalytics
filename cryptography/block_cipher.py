"""
@author: Alessio Castrica
@date: 28/02/2022
"""
from abc import ABC, abstractmethod


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
        state,
        key,
    ) -> None:
        super().__init__()
        self.state = state
        self.key = key

    @property
    def multiplication_lookup_table(self):
        pass

    @property
    def substitution_lookup_table(self):
        pass
