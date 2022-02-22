"""
@author: Alessio Castrica
@date: 07/02/2022

Python implementation di coin flipping by telephone
http://users.cms.caltech.edu/~vidick/teaching/101_crypto/Blum81_CoinFlipping.pdf

"""
import os
import random
import time

possible_choices = {"heads", "tails"}
choice = None
while choice not in possible_choices:
    os.system("cls")
    choice = str(input("Hello Alice, please enter heads or tails: ")).lower()
print(f"Alice chooses {choice}.")

# genera un numero pseudo-random (e.g. 7758176404715800194)
random.seed(0)  # per riproducibilità
n = random.randint(0, 2e20)
commitment = hash(choice) + n
print(f"Alice sent {commitment} to Bob.")

# Bob lancia la moneta
print("Bob spin the coin...")
random.seed(None)
lancio_moneta = random.randint(0, 1)
if lancio_moneta == 0:
    stato_moneta = "heads"
else:
    stato_moneta = "tails"
if hash(stato_moneta) + n == commitment:
    print(f"It's {stato_moneta}: Alice wins.")
else:
    print(f"It's {stato_moneta}: Alice loses.")
    n_primo = 0
    start_time = time.time()
    max_seconds = 10
    print(f"Alice tries to cheat, searching n' for {max_seconds} seconds!")
    while hash(stato_moneta) + n_primo != commitment:
        n_primo += 1
        current_time = time.time()
        elapsed_time = current_time - start_time

        if elapsed_time > max_seconds:
            print(
                f"Alice tried each integer from 0 to {n_primo} "
                + f"but she didn't find n' in {max_seconds} seconds."
            )
            break
        if hash(stato_moneta) + n_primo == commitment:
            print(f"Alice found n' = {n_primo}.")
