import numpy as np
# Logistic Regression

# Sigmoid Funcion
def sigmoid(theta):
    return 1 / (1 + np.exp(-theta))

# Calculate cost of logistic
def logistic_cost(W, X, Y):
    sy = np.squeeze(Y)
    theta = X.dot(W)
    hypothesis = sigmoid(theta)
    return np.mean( -(sy * (np.log(hypothesis))) - ((1-sy) * (np.log(1-hypothesis))) )

# Gradient
def logistic_gradient(W, X, Y):
    sy = np.squeeze(Y)
    theta = X.dot(W)
    hypothesis = sigmoid(theta)
    return X.T.dot(hypothesis - sy)

# Calculate Gradient Descent
def gradient_descent(W, X, Y, learning_rate=1e-3, max_step=10001):
    updatedWeight = W
    print("First Weight: %s" %(W))

    for step in range(max_step):
        cost = logistic_cost(updatedWeight, X, Y)
        updatedWeight = updatedWeight - (learning_rate * logistic_gradient(updatedWeight, X, Y))
        print("Cost: %s Weight: %s" %(cost, updatedWeight))

    return updatedWeight

# Initialize
x_data = np.array(
         [[1, 2],
          [2, 3],
          [3, 1],
          [4, 3],
          [5, 3],
          [6, 2]])
y_data = np.array(
    [[0],
     [0],
     [0],
     [1],
     [1],
     [1]])
w_data = np.zeros(x_data.shape[1])

# Debugging
print( "Logistic Cost: %s" %(logistic_cost(w_data, x_data, y_data)) )
print( "Logistic Gradient: %s" %(logistic_gradient(w_data, x_data, y_data)) )
print( "Logistic Gradient Descent: %s" %(gradient_descent(w_data, x_data, y_data, learning_rate=5e-5)))
