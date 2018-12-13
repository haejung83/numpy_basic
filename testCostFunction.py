import numpy as np
import matplotlib.pyplot as plt

# Cost function
def cost(weight, x, y):
    sum = 0
    index_arr = np.arange(0, x.size, 1)
    for index in index_arr:
        sum += (weight * x[index] - y[index]) ** 2
        print("index: %s sum: %s" %(index, sum))
    return sum / index_arr.size

# Initialize dataset
arr_x = np.arange(-10, 10, 1)
arr_y = np.arange(-10, 10, 1)

arr_weight = np.arange(-3, 5, 0.1)
arr_weight_index = np.arange(0, arr_weight.size, 1)

# Initialize result set
arr_cost = np.array(arr_weight_index)

# Iteration for testing
for weight_index in arr_weight_index:
    arr_cost[weight_index] = cost(arr_weight[weight_index], arr_x, arr_y)

print("arr_cost: %s" %(arr_cost))

plt.plot(arr_weight, arr_cost, label="cost")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Cost Function")
plt.legend()
plt.show()




