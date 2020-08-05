import numpy as np

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

class RNN:
    def __init__(self, input_size, hidden_size, batch_size = 1):
        self.grads = []
        self.W_h = np.random.randn(hidden_size, hidden_size)
        self.W_x = np.random.randn(input_size, hidden_size)
        self.b = np.random.randn(batch_size, hidden_size)
        self.params = [self.W_h, self.W_x, self.b]
        self.tanh = tanh()
        self.h_t = None
        self.h_t_minus_1 = None
        self.x = None

    def forward(self, x, h_t_minus_1):
        self.x = x
        self.h_t_minus_1 = h_t_minus_1
        self.h_t = self.tanh.forward(np.dot(x, self.W_x) + np.dot(h_t_minus_1, self.W_h) + self.b)   
        return self.h_t

    def backward(self, dnext):
        self.grads = []
        dtanh = self.tanh.backward(dnext)
        dWh = np.dot(self.h_t_minus_1.T, dtanh)
        self.grads.append(dWh)
        dWx = np.dot(self.x.T, dtanh)
        self.grads.append(dWx)
        d_h_minus_1 = np.dot(dtanh, self.W_h)
        db = dtanh.copy()
        self.grads.append(db)
        dx = np.dot(dtanh, self.W_x.T)
        return d_h_minus_1, dx

class Time_RNN:
    def __init__(self, input_size, hidden_size, batch_size, sequence_length):
        self.rnn = []
        self.h_t_list = []
        for _ in range(sequence_length):
            self.rnn.append(RNN(input_size, hidden_size, batch_size))

    def forward(self, x, h_t_minus_1):
        self.h_t_list = []
        for index, x_t in enumerate(x):
            h_t_minus_1 = self.rnn[index].forward(x_t, h_t_minus_1)
            self.h_t_list.append(h_t_minus_1)
        return self.h_t_list

    def backward(self, dnext):
        for reversed_rnn in reversed(self.rnn):
            dnext, dx = reversed_rnn.backward(dnext)
        return dnext, dx 


##h_t_minus_1 = np.random.randn(2, 3)
##x = np.array([[1, 2, 3], [1, 2, 3]])
##rnn = RNN(3, 3, 2)
##y = rnn.forward(x, h_t_minus_1)
##grads = rnn.backward(y)

##h_t_minus_1 = np.random.randn(2, 3)
##x = np.array([[[1, 2, 3], [1, 2, 3]]])
##rnn = Time_RNN(3, 3, 2, 1)
##h_last = rnn.forward(x, h_t_minus_1)
##print(h_last)
##rnn.backward(h_last)



