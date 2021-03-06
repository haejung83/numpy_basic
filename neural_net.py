import sys
import os
import numpy as np
from collections import OrderedDict

from activation import sigmoid
from identity import softmax
from loss import cross_entropy_error
from gradient import numerical_gradient_test as numerical_gradient
from layer import *



# Referenced from Deep learning from scratch

class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # Create weights
        self._params = dict()
        # First layer
        self._params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self._params['b1'] = np.zeros(hidden_size)
        # Second layer
        self._params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self._params['b2'] = np.zeros(output_size)

        # Create layers
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self._params['W1'], self._params['b1'])
        self.layers['Relu1'] = ReLU()
        self.layers['Affine2'] = Affine(self._params['W2'], self._params['b2'])
        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = dict()
        grads['W1'] = numerical_gradient(loss_W, self._params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self._params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self._params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self._params['b2'])

        return grads

    def gradient(self, x, t):
        # Forward
        self.loss(x, t)

        # Backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = dict()
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db

        return grads
