import numpy as np
import matplotlib.pyplot as plt

# Constants
max_step = 2001

# Calculate cost of matrix (manually)
def mat_cost(mat_weight, mat_x, mat_y):
    mean_sum = 0
    mean_range = mat_x.shape[0]
    for index in range(mean_range):
        weight_sum = 0
        for weight_index in range(mat_weight.size): # Iteration until size of weight dimension
            weight_sum += mat_weight[weight_index] * mat_x[index][weight_index] # Multiply
        mean_sum += (weight_sum - mat_y[index]) ** 2 # Square
    return mean_sum / mean_range

# Calculate cost of matrix (using matrix accumulation)
def mat_cost2(mat_weight, mat_x, mat_y):
    mat_mean = mat_x.dot(mat_weight).reshape(mat_x.shape[0], 1) # Multiply
    mat_mean = (mat_mean - mat_y) ** 2 # Square
    return np.sum(mat_mean / mat_x.shape[0]) # Mean value

# Gradient - Derivative
def mat_gradient(mat_weight, mat_x, mat_y):
    mat_mean = mat_x.dot(mat_weight).reshape(mat_x.shape[0], 1) # Multiply
    mat_mean = (mat_mean - mat_y) * mat_x # Square
    return np.sum(mat_mean, axis=0)

# Function: Calculate with Gradient Descent Algorithm
def mat_gradientDescent(learning_rate, mat_weight, mat_x, mat_y):
    mat_calculate_weight = mat_weight
    out_arr_cost = np.zeros(max_step)

    for index in range(max_step):
        current_cost = mat_cost2(mat_calculate_weight, mat_x, mat_y)
        out_arr_cost[index] = current_cost
        print("index: %s, cost:%s, weight:%s" % (index, current_cost, mat_calculate_weight))
        mat_calculate_weight = mat_calculate_weight-(learning_rate * mat_gradient(mat_calculate_weight, mat_x, mat_y))

    return mat_calculate_weight, out_arr_cost

# Initialize Variables
W = np.array([1., 2., 3.]) # 3x1
X = np.array([[73., 80., 75.], [93., 88., 93.], [89., 91., 90.], [96., 98. ,100.], [73., 66., 70.]]) # 5x3
Y = np.array([[152.], [185.], [180], [196.], [142.]]) # 5x1

# Debugging
print("W: %s shape:%s" %(W, W.shape))
print("X: %s shape:%s" %(X, X.shape))
print("Y: %s shape:%s" %(Y, Y.shape))

print("Mat Cost: %s" %(mat_cost(W, X, Y)))
print("Mat Cost2: %s" %(mat_cost2(W, X, Y)))
print("Mat Gradient: \n%s" %(mat_gradient(W, X, Y)))
print("Mat Multiply: \n%s" %(W * X))

print("Gradient Descent with Matrix")
cal_weight, arr_cost = mat_gradientDescent(1e-5, W, X, Y)
print("Weight: %s" %(cal_weight))

plt.plot(range(max_step), arr_cost, label="Cost")
plt.xlabel("step")
plt.ylabel("cost")
plt.title("Gradient Descent with Matrix")
plt.legend()
plt.show()

