import numpy as np


class Dropout:
    def __init__(self, dropout_ratio = 0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x):
        self.mask = np.zeros_like(x)
        number_of_sample = x.shape[0]
        number_of_activation = int(number_of_sample * self.dropout_ratio)     
        activated_index = np.random.randint(number_of_sample, size=number_of_activation)
        self.mask[activated_index] = np.array(np.ones(x.shape[1]))
        return self.mask * x

    def backward(self, dout):
        return dout * self.mask

        
        
##a = np.array([[0, 1], [1, 2], [2, 3], [2, 3]])
##up_grad = np.array([[1, 1], [1, 1], [1, 1], [1, 1]])
##dropout = Dropout()
##y = dropout.forward(a)
##k = dropout.backward(up_grad)
##print(y)
##print(k)
