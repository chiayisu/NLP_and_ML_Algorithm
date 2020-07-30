import numpy as np

def CrossEntropyLoss(y, y_hat):
    N = y.shape[0]
    log_probability = np.log(y_hat)
    total_loss = np.sum(y * log_probability)
    total_loss = total_loss * -1 
    return total_loss / N

class SoftmaxwithLoss:
    def __init__(self):
        self.params = []
        self.y_hat = None
        self.y = None
    
    def forward(self, x, y):
        y_hat = Softmax(x)
        self.y_hat = y_hat
        loss = CrossEntropyLoss(y, y_hat)
        return loss

    def backward(self, dout=1):
        dimension = self.y_hat.shape[0]
        self.y_hat[y==1] -= 1
        dx = self.y_hat * dout
        dx /= dimension
        return dx
