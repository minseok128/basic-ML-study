import numpy as np
import matplotlib.pyplot as plt

# def AND(x1, x2):
#     w1, w2, theta = 0.5, 0.5, 0.7
#     tmp = w1 * x1 + w2 * x2
#     if tmp > theta:
#         return 1
#     else:
#         return 0

def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.dot(x, w) + b
    if tmp > 0:
        return 1
    else:
        return 0

print(AND(0, 0))
print(AND(0, 1))
print(AND(1, 0))
print(AND(1, 1))
print()

def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.dot(x, w) + b
    if tmp > 0:
        return 1
    else:
        return 0

print(NAND(0, 0))
print(NAND(0, 1))
print(NAND(1, 0))
print(NAND(1, 1))
print()

def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([1, 1])
    b = -0.7
    tmp = np.dot(x, w) + b
    if tmp > 0:
        return 1
    else:
        return 0
    
print(OR(0, 0))
print(OR(0, 1))
print(OR(1, 0))
print(OR(1, 1))
print()

def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    return AND(s1, s2)

print(XOR(0, 0))
print(XOR(0, 1))
print(XOR(1, 0))
print(XOR(1, 1))
print()