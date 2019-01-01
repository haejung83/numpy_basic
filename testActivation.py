from activation import sigmoid, step, relu
import numpy as np
import matplotlib.pylab as plt

# Input data
x = np.arange(-10, 10, 0.1)

print('Sigmoid Function')
y = sigmoid(x)
plt.plot(x, y)
plt.show()
print(y)

print('Step Function')
y = step(x)
plt.plot(x, y)
plt.show()
print(y)

print('Relu Function')
y = relu(x)
plt.plot(x, y)
plt.show()
print(y)