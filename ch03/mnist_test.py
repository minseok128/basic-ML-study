import sys, os, pickle
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.pardir)
from dataset.mnist import load_mnist

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    c = np.max(x)
    exp_a = np.exp(x - c)
    return (exp_a / np.sum(exp_a))

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True)
    return x_test, t_test

def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = x @ W1 + b1
    z1 = sigmoid(a1)
    a2 = z1 @ W2 + b2
    z2 = sigmoid(a2)
    a3 = z2 @ W3 + b3
    y = softmax(a3)
    return y

network = init_network()
x, t = get_data()

# accuracy_cnt = 0
# for i in range(len(x)):
#     y = predict(network, x[i])
#     p = np.argmax(y)
#     if p == t[i]:
#         accuracy_cnt += 1
# print("Accuracy: ", float(accuracy_cnt / len(x)))

batch_size = 100
accuracy_cnt = 0

for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print("Accuracy: ", float(accuracy_cnt / len(x)))