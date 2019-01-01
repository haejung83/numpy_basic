import numpy as np

def identity(x):
    reutrn x

def softmax(x):
    xmax = np.max(x) # for preventing overflow
    xnexp = np.exp(x-xmax)
    return xnexp / np.sum(xnexp)