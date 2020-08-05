import numpy as np


## for Output Layer
def Softmax(x):
    denominator = np.sum(np.exp(x), axis = 1)
    denominator = denominator[:, np.newaxis]
    return np.exp(x) / np.exp(denominator)

class Sigmoid:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x))
        return 1 / (1 + np.exp(-x))
    
    def backward(self, dout):
        return dout * self.out * (1 - self.out)

class Relu:
    def __init__(self):
        self.params, self.grads = [], []
        self.x = None
    
    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, dout):
        dout[self.x <= 0] = 0
        return dout

class tanh:
    def __init__(self):
        self.params, self.grads = [], []
        self.output = None
    
    def forward(self, x):
        numerator = np.exp(x) - np.exp(-x)
        denominator = np.exp(x) + np.exp(-x)
        self.output = numerator / denominator
        return self.output

    def backward(self, dout):
        dout = dout * (1 - np.square(self.output))
        return dout
