"""
@author: Alessio Castrica
@date: 22/02/2022

Implementazione in python del cifrario A5/1.

Ho seguito il paper https://www.rocq.inria.fr/secret/Anne.Canteaut/encyclopedia.pdf 
senza implementare nè la fase di warm-up nè i clock irregolari.

Il keystream è generato facendo lo XOR tra il key bit e i feedback bit dei 3 LSFR. 
Se la rappresentazione binaria del messaggio è più lunga della somma tra frame_number e secrecy_key
allora per i bit eccedenti lo XOR è operato solamente tra i feedback bit dei 3 LSFR.
"""

from abc import ABC, abstractmethod
import re
from typing import Dict, List, Optional, Union
import random


def validate_binary_list(
    list_of_numbers: List[int],
) -> bool:
    """
    Takes a list of bits and returns true if it's made only of 0s and 1s.
    Parameters
    ----------
    list_of_numbers: List[int]
        A list of 0s and 1s.

    Returns
    -------
    A boolean value, true if the list is made only of 0s and 1s.
    Otherwise it raises value errors.
    """
    validation = False
    if isinstance(list_of_numbers, List):
        for index, value in enumerate(list_of_numbers):
            if isinstance(value, int) and (value == 1 or value == 0):
                validation = True
            else:
                raise ValueError(
                    "All values in the list must be 1s or 0s, "
                    + f"while at index {index} the value is {value}."
                )
    else:
        raise ValueError("The argument must be a list.")
    return validation


def text_to_bits(
    text: str,
    encoding: str = "utf-8",
    errors: str = "surrogatepass",
) -> List[int]:
    """
    Takes a string and returns it's binary representation.

    Parameters
    ----------
    text: str
        Any string.

    Returns
    -------
    A list of 0s and 1s.
    """
    bits = bin(int.from_bytes(text.encode(encoding, errors), "big"))[2:]
    bits_list = []
    for bit in bits.zfill(8 * ((len(bits) + 7) // 8)):
        bits_list.append(int(bit))
    return bits_list


def text_from_bits(
    bits_list: List[int],
    encoding: str = "utf-8",
    errors: str = "surrogatepass",
) -> str:
    """
    Takes a list of bits and returns it's text message.

    Parameters
    ----------
    bits_list: List[int]
        A list of 0s and 1s.

    Returns
    -------
    A string.
    """
    assert validate_binary_list(bits_list)
    string_list_bits = [str(bit) for bit in bits_list]
    str_of_bits = "".join(string_list_bits)
    n = int(str_of_bits, 2)
    return n.to_bytes((n.bit_length() + 7) // 8, "big").decode(encoding, errors) or "\0"


def binary_list_to_string(
    bits_list: List[int],
) -> str:
    """
    Takes a list of bits and returns it's text message.

    Parameters
    ----------
    bits_list: List[int]
        A list of 0s and 1s.

    Returns
    -------
    A string with 0s and 1s.
    """
    assert validate_binary_list(bits_list)
    return "".join([str(bit) for bit in bits_list])


class LinearFeedbackShiftRegisters:
    """
    Linear Feedback Shift Registers (LFSRs) are the basic components of many
    running-key generators for stream cipher applications,
    because they are appropriate to hardware implementation and
    they produce sequences with good statistical properties.
    LFSR refers to a feedback shift register with a linear feedback function.

    Parameters
    ----------
    length: int
        The length of the LSFR object.
    initial_state: Optional[List[int]] = None
        The initial state of the LSFR object.
    taps: Optional[List[int]]
        The taps of the LSFR object.
    """

    def __init__(
        self,
        length: int,
        initial_state: Optional[List[int]] = None,
        taps: Optional[List[int]] = None,
    ) -> None:
        self.length = length
        if initial_state is None or validate_binary_list(initial_state):
            self._initial_state = initial_state
        self._taps = taps
        self._state = None

    @property
    def initial_state(self) -> List[int]:
        if self._initial_state is None:
            self._initial_state = [random.randint(0, 1) for _ in range(self.length)]
        return self._initial_state

    @initial_state.setter
    def initial_state(self, init_st: List[int]) -> None:
        if isinstance(init_st, List):
            for _ in init_st:
                if isinstance(_, int) and (_ == 1 or _ == 0):
                    self._initial_state = init_st
                else:
                    raise ValueError(
                        "All values in the initial state list must be 1s or 0s,"
                        + f"while {_} the value is not."
                    )
        else:
            raise ValueError("The initial state must be a list.")
        assert (
            len(self._initial_state) == self.length
        ), f"The initial state must have the same length as the overall {self.__class__.__name__} object."

    @property
    def taps(self) -> List[int]:
        """Indeces of states that we take for the update."""
        if self._taps is None:
            self._taps = []
        return self._taps

    @taps.setter
    def taps(self, new_taps: List[int]) -> None:
        if isinstance(new_taps, List):
            for _ in new_taps:
                if isinstance(_, int):
                    self._taps = new_taps
                else:
                    raise ValueError(
                        "All values in the taps list must be integers,"
                        + f"while {_} is not."
                    )
        else:
            raise ValueError("Taps must be a list.")

    @property
    def state(self) -> List[int]:
        if self._state is None:
            self._state = self.initial_state
        return self._state

    @state.setter
    def state(self, new_state: List[int]):
        if validate_binary_list(new_state):
            self._state = new_state
        assert (
            len(self._state) == self.length
        ), f"The state must have the same length as the overall {self.__class__.__name__} object."

    def update(self, n_cicles: int = 1) -> Dict[int, int]:
        """
        Update the state of the LSFR object as many times as n_cicles.

        Parameters
        ----------
        n_cicles: int
            Number of times the user wants to update the state.

        Returns
        -------
        feedback_bits: Dict[int, int]
            The feedback bit at each iteration.
            The key is the number of the iteration while the value is the feedback bit.
        """
        self.states_in_time = {0: self.state}
        feedback_bits: Dict[int, int] = {}
        for cicle in range(n_cicles):
            # insert sum of bits
            feedback_bit = sum([self.state[i] for i in self.taps]) % 2
            self.state.insert(0, feedback_bit)
            # remove last bit
            self.state.pop()
            self.states_in_time[cicle + 1] = self.state
            feedback_bits[cicle] = feedback_bit
        return feedback_bits


class StreamCipher(ABC):
    """
    Abstract class that represents a stream cipher.
    A stream cipher is a symmetric cipher which operates with a time-varying transformation on
    individual plaintext digits.
    """

    @abstractmethod
    def encrypt(self, plaintext: Union[List[int], str]) -> List[int]:
        pass

    @abstractmethod
    def decrypt(self, ciphertext: List[int]) -> List[Union[int, str]]:
        pass


class A5_1(StreamCipher):
    """
    A5/1 is the symmetric cipher used for encrypting over-the-air
    transmissions in the GSM standard.

    Parameters
    ----------
    secrecy_key: List[int]
        A user defined key, default is a random 64-bit key.
    frame_number: List[int]
        A public frame number, default is a random 22-bit key.
    """

    def __init__(
        self,
        secrecy_key: List[int] = [random.randint(0, 1) for _ in range(64)],
        frame_number: List[int] = [random.randint(0, 1) for _ in range(22)],
    ) -> None:
        super().__init__()
        if validate_binary_list(secrecy_key) and len(secrecy_key) == 64:
            self.secrecy_key = secrecy_key
        else:
            raise ValueError("The key must be a 64-bit list of 1s and 0s")
        if validate_binary_list(frame_number) and len(frame_number) == 22:
            self.frame_number = frame_number
        else:
            raise ValueError("The frame number must be a 22-bit list of 1s and 0s")
        lsfr1 = LinearFeedbackShiftRegisters(
            length=19, taps=[13, 16, 17, 18], initial_state=self.secrecy_key[0:19]
        )
        lsfr2 = LinearFeedbackShiftRegisters(
            length=22, taps=[20, 21], initial_state=self.secrecy_key[0:22]
        )
        lsfr3 = LinearFeedbackShiftRegisters(
            length=23, taps=[7, 20, 21, 22], initial_state=self.secrecy_key[0:23]
        )
        self.set_of_lsfrs = {lsfr1, lsfr2, lsfr3}

    def get_key_from_user_input(self) -> None:
        """Get secrecy key from user input in the terminal."""
        user_key = ""
        while len(user_key) != 64 or not re.match("^([01])+", user_key):
            user_key = str(input("Please enter a 64-bit key: "))
            if len(user_key) == 64 and re.match("^([01])+", user_key):
                self.secrecy_key = [int(bit) for bit in user_key]

    def generate_keystream(self, binary_messsage_represenation: List[int]) -> List[int]:
        """
        The keystream is generated by xored the key bit Kt to the feedback bit of each LSFR.
        """
        # TODO: da rivedere!
        generator_initial_state = self.secrecy_key + self.frame_number
        lsfr_feedback_bits = {}
        for index, lsfr in enumerate(self.set_of_lsfrs):
            lsfr_feedback_bits[index] = list(
                lsfr.update(n_cicles=len(binary_messsage_represenation)).values()
            )
        keystream = []
        if len(binary_messsage_represenation) <= len(generator_initial_state):
            for bit in range(len(binary_messsage_represenation)):
                keystream.append(
                    generator_initial_state[bit]
                    ^ lsfr_feedback_bits[0][bit]
                    ^ lsfr_feedback_bits[1][bit]
                    ^ lsfr_feedback_bits[2][bit]
                )
        else:
            for bit in range(len(generator_initial_state)):
                keystream.append(
                    generator_initial_state[bit]
                    ^ lsfr_feedback_bits[0][bit]
                    ^ lsfr_feedback_bits[1][bit]
                    ^ lsfr_feedback_bits[2][bit]
                )
            for bit in range(
                len(binary_messsage_represenation) - len(generator_initial_state)
            ):
                keystream.append(
                    lsfr_feedback_bits[0][bit]
                    ^ lsfr_feedback_bits[1][bit]
                    ^ lsfr_feedback_bits[2][bit]
                )
        self.keystream = keystream
        return self.keystream

    def encrypt(self, plaintext: Union[List[int], str]) -> List[int]:
        if isinstance(plaintext, str):
            binary_representation = text_to_bits(plaintext)
        elif isinstance(plaintext, list):
            if validate_binary_list(plaintext):
                binary_representation = plaintext
        else:
            raise ValueError("Plaintext must be a string or a list of 0s and 1s.")
        self.keystream = self.generate_keystream(binary_representation)
        ciphertext = []
        for bit in range(len(binary_representation)):
            ciphertext.append(self.keystream[bit] ^ binary_representation[bit])
        return ciphertext

    def decrypt(self, ciphertext: List[int]) -> List[Union[int, str]]:
        plaintext = []
        for bit in range(len(ciphertext)):
            plaintext.append(self.keystream[bit] ^ ciphertext[bit])
        return text_from_bits(plaintext)


if __name__ == "__main__":

    ## UNCOMMENT THIS LINES IF YOU WANT TO TEST THE IMPLEMENTATION OF THE LSFR ##
    # print("\033[1mSingle LSFR implementation\033[0m:")
    # lsfr = LinearFeedbackShiftRegisters(
    #     length=19,
    #     taps=[13, 16, 17, 18],
    # )
    # print(f"The initial state of the LSFR is: \n{lsfr.state} ")
    # print(f"The taps (0-based indexing) of the LSFR are: \n{lsfr.taps}\n ")
    # n_updates = 3
    # print(
    #     f"The feedback bits of the LSFR when updating {n_updates} times are: \n{lsfr.update(n_updates)}\n "
    # )
    # print(f"The state after the update of the LSFR is: \n{lsfr.state}\n ")
    # print(f"The states along time of the LSFR are: \n{lsfr.states_in_time}\n ")

    print("\033[1mA5/1 implementation\033[0m\n")
    a5_1 = A5_1()

    ## UNCOMMENT THIS LINES IF YOU WANT TO PROVIDE A CUSTOM 64-BIT KEY ##
    # print(
    #     f"Default random 64-bit secrecy key: {binary_list_to_string(a5_1.secrecy_key)}"
    # )
    # print("The user can provide a custom key.")
    # a5_1.get_key_from_user_input()
    # print("If provided the secrecy key of the object will be updated:")
    # print(a5_1.secrecy_key)

    # Può inserire il messaggio sia nella variabile sottostante che nel terminal
    message = ""
    while len(message) == 0:
        message = str(input("Please enter a message: "))
    print(f"Message: {message}")
    print(f"Message in bits: \n{text_to_bits(message)}")
    print(f"Length of message in bits: {len(text_to_bits(message))}")
    print("\nKeystream:")
    print(a5_1.generate_keystream(text_to_bits(message)))
    print("Length: " + str(len(a5_1.generate_keystream(text_to_bits(message)))))
    enc_message = a5_1.encrypt(plaintext=str(message))
    print(f"\n\033[1mEncrypted message\033[0m: \n{enc_message}")
    print(f"Length encrypted message: \n{len(enc_message)}")
    dec_message = a5_1.decrypt(ciphertext=enc_message)
    print(f"\n\033[1mDecrypted message\033[0m: \n{dec_message}")
    assert dec_message == message
