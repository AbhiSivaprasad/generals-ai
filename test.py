import random

N = 10
target = random.randrange(0, 2**N)
noise_rate = 0

def xor_sum(value):
    noise = sum([2 ** i for i in range(N) if random.random() < noise_rate])
    return bin(target ^ value ^ noise).count('1')

print(bin(target))
print(xor_sum(0))

