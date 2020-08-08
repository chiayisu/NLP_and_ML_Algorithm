import numpy as np

class Affine:
    def __init__(self, input_size, hidden_size, batch_size):
        self.W = np.random.randn(input_size, hidden_size)
        self.b =np.random.randn(batch_size, hidden_size)
        self.params = [self.W, self.b]
        self.grads = [np.zeros_like(self.W), np.zeros_like(self.b)]
        self.x = None

    def forward(self, x):
        out = np.dot(x, self.W) + self.b
        self.x = x
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        dW = np.dot(self.x.T, dout)
        db = dout.copy()
        self.grads[0] = dW.copy()
        self.grads[1] = db.copy()
        return dx
