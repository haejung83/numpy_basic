import numpy as np
from activation import sigmoid

X = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])

print(X.shape)
print(W1.shape)
print(B1.shape)

A1 = np.dot(X, W1) + B1
print('A1: {}'.format(A1))

Z1 = sigmoid(A1)
print('sigmoid(A1): {}'.format(Z1))

W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])

print(Z1.shape)
print(W2.shape)
print(B2.shape)

A2 = np.dot(Z1, W2) + B2
print('A2: {}'.format(A2))

Z2 = sigmoid(A2)
print('sigmoid(A2): {}'.format(Z2))