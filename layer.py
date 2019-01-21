import numpy as np

class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0 # mask에서 True인 위치의 요소만 가져와 배열로 만든다.
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx