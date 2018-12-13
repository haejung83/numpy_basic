import numpy as np
import matplotlib.pyplot as plt

# Constants
max_step = 10001

# Sigmoid Function (Hypothesis)
def sigmoid(x):
    return 1 / (1 + (np.exp(-x)))

# Calculate a cost of logistic regression
def costLogisticRegression(weight, in_x, in_y):
    mean_h = 0
    in_x = in_x * weight
    for step in range(in_x.size):
        hypothesis = sigmoid(in_x[step])
        mean_h += (in_y[step]*np.log(hypothesis)) + ((1-in_y[step])*np.log(1-hypothesis))
    return mean_h / in_x.size

def gradientDescent(learning_rate, weight, arr_x, arr_y):
    var_weight = weight
    for index in range(max_step):
        print("index: %s, weight:%s" % (index, var_weight))
        var_weight = var_weight-(learning_rate * costLogisticRegression(var_weight, arr_x, arr_y))
    return var_weight

# Initialize variables
val_input_x = np.arange(-5, 5, 1)
val_input_y = np.array([0,0,0,0,0,1,1,1,1,1])
val_sigmoid = np.zeros(val_input_x.size)

# Debugging - Calculation
for step in range(val_input_x.size):
    val_sigmoid[step] = sigmoid(val_input_x[step])

val_cost = costLogisticRegression(1, val_input_x, val_input_y)
print("Cost: %s" %(val_cost))

initial_weight = 1
print("Gradient Descent Algorithm: %s" %(gradientDescent(0.01, initial_weight, val_input_x, val_input_y)))

# Draw a plot
plt.plot(val_input_x, val_sigmoid, label='sigmoid')
plt.title('Simple Sigmoid')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

