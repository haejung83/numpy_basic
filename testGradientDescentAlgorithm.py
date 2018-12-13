import numpy as np
import matplotlib.pyplot as plt

# Constants
max_step = 100

# Function: Calculate derivative cost
def costDerivative(weight, arr_x, arr_y):
    sum = 0
    for index in range(arr_x.size):
        sum += (weight * arr_x[index] - arr_y[index]) * arr_x[index]
        #print("index: %s sum: %s" %(index, sum))
    return sum / arr_x.size

# Function: Calculate with Gradient Descent Algorithm
def gradientDescent(learning_rate, weight, arr_x, arr_y):
    var_weight = weight
    updatedWeight = weight
    for index in range(max_step):
        print("index: %s, weight:%s" % (index, var_weight))
        var_weight = var_weight-(learning_rate * costDerivative(var_weight, arr_x, arr_y))
        if (updatedWeight-var_weight == 0):
            break
        else:
            updatedWeight = var_weight
    return var_weight

# Initialize Data
arr_x = np.arange(1, 4, 1)
arr_y = np.arange(1, 4, 1)

#y_start = 1
#y_end = 4
#y_step = abs(y_end-y_start)/arr_x.size
#arr_y = np.arange(y_start, y_end, y_step)

learning_rate = 0.1
initial_weight = -3.0

print("Arr_X:%s" %(arr_x))
print("Arr_Y:%s" %(arr_y))
print("Arr Size:%s" %(arr_x.size))

arr_index = np.arange(0, arr_x.size, 1)

calculatedWeight = gradientDescent(learning_rate, initial_weight, arr_x, arr_y)
print("Calculated Weight: %s" %(calculatedWeight))

expectedY = arr_x * calculatedWeight

plt.plot(arr_index, arr_x, label="Array X")
plt.plot(arr_index, arr_y, label="Array Y")
plt.plot(arr_index, expectedY, label="Expected Y")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Gradient Descent")
plt.legend()
plt.show()

