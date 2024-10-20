import random
import binstr

random.seed(binstr.str_to_b(input("Input Seed 1: ")))
for i in range(5):
    print(random.random())
random.seed(binstr.str_to_b(input("Input Seed 2: ")))
for i in range(5):
    print(random.random())