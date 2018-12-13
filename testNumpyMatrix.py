import numpy as np

X = np.array([[73., 80., 75.], [93., 88., 93.], [89., 91., 90.], [96., 98. ,100.], [73., 66., 70.]]) # 5x3
W = np.array([1., 2., 3.]) # 3x1

Y = np.array([[152.], [185.], [180], [196.], [142.]]) # 5x1

#print("A1: %s" %(X*W))
#print("A2: %s" %(X.dot(W)))
#print("Y1: %s, Y2: %s" %(X, np.transpose(X)))
#print("Y1: %s, Y2: %s" %(X.shape, np.transpose(X).shape))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

print("B1: %s" %(sigmoid(X.dot(W))))
print("B2: %s, %s" %(np.transpose(X).shape, sigmoid(X.dot(W)).shape))
print("B3: %s" %( np.transpose(X).dot(sigmoid(X.dot(W))) ))

A = 0.01
M = X.shape[0]

print("M1: %s" %(M))

for step in range(100):
    print("W: %s" %(W))
    W = W - (A * ( np.transpose(X).dot(sigmoid(X.dot(W)) ))) / M

print("END-W: %s" %(W))


TX = np.zeros([2,3,4])
print("TX: \n%s" %(TX))
print("NDIM: %s" %(TX.ndim))
