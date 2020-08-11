import numpy as np


def Sigmoid(x):
    out = 1 / (1 + np.exp(-x))
    return 1 / (1 + np.exp(-x))

class SGD:
    def __init__(self, lr = 0.01):
        self.lr = lr

    def update(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.lr * grads[i] 

class CBOW_with_Negative_Sampling:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.grads, self.params = [], []
        self.hidden_dim = hidden_dim
        self.Win = np.random.randn(input_dim, hidden_dim)
        self.Wout = np.random.randn(hidden_dim, output_dim)
        self.params.append(self.Wout)
        self.params.append(self.Win)
        self.center = None
        self.center_index = None
        self.pos_output = None
        self.neg_output = None
        self.h = None
        self.negative_sample = None
        self.neg_list = []
        
    def forward(self, center_word, Negative_Sample):
        h = np.dot(center_word, self.Win)
        self.h  = h
        self.center_index = np.argmax(center_word)
        hidden_1 = np.dot(self.Wout.T[self.center_index].reshape(1,self.hidden_dim), self.h.T)
        self.pos_output = Sigmoid(hidden_1)
        loss = np.log(self.pos_output) * (-1)
        
##        Negtive Sampling
        Wout_T = self.Wout.T
        self.negative_sample = Negative_Sample
        for negative_sample in Negative_Sample:
            negative = np.dot(Wout_T[negative_sample].reshape(1, self.hidden_dim), self.h.T)
            negative = negative * (-1)
            self.neg_output = Sigmoid(negative)
            self.neg_list.append(self.neg_output)
            loss += np.log(self.neg_output) * (-1)
        return loss[0]

    def backward(self):
        self.grads = []
        dWout = np.zeros_like(self.Wout.T)
        dWout[self.center_index] = (self.pos_output - 1) * self.h
        for index, negative_sample in enumerate(self.negative_sample):
            dWout[negative_sample] = self.neg_list[index] * self.h
        self.grads.append(dWout.T)

        dWin = np.zeros_like(self.Win)

        temp = (self.pos_output - 1) * dWin[self.center_index].reshape(1, self.hidden_dim) 
        for index, negative_sample in enumerate(self.negative_sample):
            temp += self.neg_list[index] * self.h
        dWin[self.center_index] = temp
        self.grads.append(dWin)

context = np.array([[0, 1, 0, 1, 0]])
negtive_Sample = np.array([0, 2])
center_word = np.array([[0, 0, 0, 0, 1]])
Word_Vector = CBOW_with_Negative_Sampling(5, 3, 5)
optimizer = SGD(0.0001)
loss_list = []
for epoch in range(10):
    loss = Word_Vector.forward(center_word, negtive_Sample)
    loss_list.append(loss)
    dout = Word_Vector.backward()
    optimizer.update(Word_Vector.params, Word_Vector.grads)
print(loss_list)
