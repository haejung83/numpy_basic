import sys
import os
import numpy as np

from activation import sigmoid
from identity import softmax
from loss import cross_entropy_error
from gradient import numerical_gradient_batch as ng

# Referenced from Deep learning from scratch

class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self._params = dict()
        # First layer
        self._params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self._params['b1'] = np.zeros(hidden_size)

        # Second layer
        self._params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self._params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self._params['W1'], self._params['W2']
        b1, b2 = self._params['b1'], self._params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        z2 = sigmoid(a2)

        y = softmax(z2)
        return y

    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = dict()
        grads['W1'] = ng(loss_W, self._params['W1'])
        grads['b1'] = ng(loss_W, self._params['b1'])
        grads['W2'] = ng(loss_W, self._params['W2'])
        grads['b2'] = ng(loss_W, self._params['b2'])

        return grads
