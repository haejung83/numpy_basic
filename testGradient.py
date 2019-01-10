from gradient import numerical_gradient, gradient_descent
import numpy as np

def function_square(x):
    return x[0]**2 + x[1]**2

xarr = np.array([[3.0, 4.0], [0.0, 0.2], [3.0, 0.0]])
lr = 0.1
step_num = 5

for x in xarr:
    grad = numerical_gradient(function_square, x)
    dx = gradient_descent(function_square, x.copy(), lr=lr, step_num=step_num)
    print('x: {}, grad: {}, dx: {}'.format(x, grad, dx))
