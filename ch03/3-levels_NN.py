import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def identity(x):
	return x

# X = np.array([1.0, 0.5])
# W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
# B1 = np.array([0.1, 0.2, 0.3])
# A1 = X @ W1 + B1
# Z1 = sigmoid(A1)
# # print(Z1)

# W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
# B2 = np.array([0.1, 0.2])
# A2 = Z1 @ W2 + B2
# Z2 = sigmoid(A2)
# # print(Z2)

# W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
# B3 = np.array([0.1, 0.2])
# A3 = Z2 @ W3 + B3
# Z3 = identity(A3)
# print(Z3)

def init_network():
    network = {}
    network['W'] = {}
    network['B'] = {}
    network['W'][0] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['B'][0] = np.array([0.1, 0.2, 0.3])
    network['W'][1] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['B'][1] = np.array([0.1, 0.2])
    network['W'][2] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['B'][2] = np.array([0.1, 0.2])
    return network

def forward(network, x):
    max_level = len(network['W'])
    y = x
    i = 0
    while i != max_level - 1:
        w = network['W'][i]
        b = network['B'][i]
        y = sigmoid(y @ w + b)
        i += 1
    y = identity(y @ network['W'][max_level - 1] + network['B'][max_level - 1])
    return y

network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y)