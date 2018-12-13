import numpy as np
import matplotlib.pyplot as plt

# SoftMax
def softmax(a):
    a_max = np.max(a) # Preventing overflow when getting value with exponential
    a_exp = np.exp(a-a_max)
    return a_exp / np.sum(a_exp)

def cost_softmax(W, X, Y):
    logits = X.dot(W)
    hypothesis = softmax(logits)
    return np.mean(-np.sum(Y * np.log(hypothesis), axis=1))

def make_one_hot(labels, classes_size):
    labels = labels.squeeze().astype(np.int32)
    one_hot = np.zeros([labels.size, classes_size])
    for step in range(one_hot.shape[0]):
        one_hot[step][labels[step]] = 1.
    return one_hot

np.random.seed(1063567050)
xy = np.loadtxt('data-04-zoo.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

nb_classes = 7

# w_data = np.random.rand(16, 1)
w_data = np.random.rand(16, nb_classes)

one_hot_ydata = make_one_hot(y_data, nb_classes)

print("Xdata--")
print(x_data.shape)
print("Ydata--")
print(y_data.shape)
print("Wdata--")
print(w_data.shape)

#print("one hot Ydata--")
#print(one_hot_ydata)

#sX = softmax(x_data)
#print("Softmax(X): %s - Sum:%s" %(sX, np.sum(sX)))

costX = cost_softmax(w_data, x_data, y_data)
print("cost Softmax(X): %s" %(costX))
