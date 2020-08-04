import numpy as np

class Embedding_Layer:

    def __init__(self, embbeded_matrix):
        self.embedded_weight = embbeded_matrix
        self.index = None
        self.params = self.embedded_weight.copy()
        self.grads = np.zeros_like(self.params)

    def forward(self, index):
        self.index = index
        return self.embedded_weight[index]

    def backward(self, dout):
        np.add.at(self.grads, self.index, dout)

a = np.array([[0, 1], [1, 2], [2, 3]])
dout = np.array([[0, 1], [1, 2]])
embbed_layer = Embedding_Layer(a)

output = embbed_layer.forward([0, 2])
embbed_layer.backward(dout)
print(output)
print()
print(embbed_layer.grads)
