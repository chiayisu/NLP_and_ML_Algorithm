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

class Sigmoid:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x))
        return 1 / (1 + np.exp(-x))
    
    def backward(self, dout):
        return dout * self.out * (1 - self.out)

class LSTM:
    def __init__(self, Wh, Wx, b):
        self.grads = []
        self.hidden_size = Wh.shape[0]
        self.W_h = Wh.copy()
        self.W_x = Wx.copy()
        self.b = b.copy()
        self.params = [self.W_x, self.W_h, self.b]
        self.c_t = None
        self.h_t = None
        self.A = None
        self.forget_sigmoid = Sigmoid()
        self.new_info_tanh = tanh()
        self.information_sigmoid = Sigmoid()
        self.output_sigmoid = Sigmoid()
        self.cell_tanh = tanh()
        self.forget_gate = None
        self.new_info = None
        self.info_gate = None
        self.output_gate = None
        self.out_value = None
        self.c_minus_1 = None
        self.h_minus_1 = None
        self.x = None

    def forward(self, x, h_t_minus_1, c_minus_1):
        self.x = x
        self.c_minus_1 = c_minus_1
        self.h_minus_1 = h_t_minus_1
        self.A = np.dot(x, self.W_x) + np.dot(h_t_minus_1, self.W_h) + self.b
        forget_index = self.hidden_size
        new_information_index = 2 * self.hidden_size
        input_index = 3 * self.hidden_size
        self.forget_gate = self.forget_sigmoid.forward(self.A[:, :forget_index])
        self.new_info = self.new_info_tanh.forward(self.A[:, forget_index:new_information_index])
        self.info_gate = self.information_sigmoid.forward(self.new_info)
        self.output_gate = self.output_sigmoid.forward(self.A[:, input_index:])
        self.c_t = (self.forget_gate * c_minus_1 +
        self.new_info * self.info_gate)
        self.out_value = self.cell_tanh.forward(self.c_t)
        self.h_t =  self.output_gate * self.out_value
        return self.h_t, self.c_t

    def backward(self, dhnext, dcnext):
        self.grads = []
        do = dhnext * self.out_value
        dcnext += self.output_gate * self.cell_tanh.backward(dhnext)
        dcprev = self.forget_gate * dcnext
        df = self.c_minus_1 * dcnext
        dg = self.info_gate * dcnext
        di = self.new_info * dcnext
        df *= self.forget_sigmoid.backward(1)
        dg *= self.new_info_tanh.backward(1)
        di *= self.information_sigmoid.backward(1)
        do *= self.output_sigmoid.backward(1)
        dA = np.hstack((df, dg, di, do))
        dWx = np.dot(self.x.T, dA)
        dWh = np.dot(self.h_minus_1.T, dA)
        db = dA.copy()
        self.grads = [dWx, dWh, db]
        dhprev = np.dot(dA, self.W_h.T)
        dx = np.dot(dA, self.W_x.T)
        
        return dcprev, dhprev, dx

class Time_LSTM:
    def __init__(self, Wh, Wx, b, sequence_length):
        self.lstm = []
        self.params = [Wh, Wx, b]
        self.grads = [np.zeros_like(Wh), np.zeros_like(Wx), np.zeros_like(b)]
        self.h_t_list = []
        self.c_t_list = []
        for _ in range(sequence_length):
            self.lstm.append(LSTM(Wh, Wx, b))
            
    def forward(self, x, h_t_minus_1, c_minus_1):
        self.h_t_list = []
        self.c_t_list = []
        for index, x_t in enumerate(x):
            h_t_minus_1, c_t_minus_1 = self.lstm[index].forward(x_t, h_t_minus_1, c_minus_1)
            self.h_t_list.append(h_t_minus_1)
            self.c_t_list.append(c_t_minus_1)
        return self.h_t_list, self.c_t_list

    def backward(self, dhnext, dcnext):
        Wh, Wx, b = self.params
        dx_list = []
        dcnext_list = []
        dhnext_list = []
        self.grads = [np.zeros_like(Wh), np.zeros_like(Wx), np.zeros_like(b)]
        for reversed_rnn in reversed(self.lstm):
            dcnext, dhnext, dx = reversed_rnn.backward(dhnext, dcnext)
            self.grads += reversed_rnn.grads
            dx_list.append(dx)
            dcnext_list.append(dcnext)
            dhnext_list.append(dhnext)
        return dcnext_list, dhnext_list, dx_list        
        
##class SGD:
##    def __init__(self, lr = 0.01):
##        self.lr = lr
##
##    def update(self, params, grads):
##        for i in range(len(params)):
##            params[i] -= self.lr * grads[i]

##input_size = 3
##hidden_size = 3
##batch_size = 2
##
##W_h = np.random.randn(hidden_size, 4 * hidden_size)
##W_x = np.random.randn(input_size, 4 * hidden_size)
##b = np.random.randn(batch_size, 4 * hidden_size)
##h_t_minus_1 = np.random.randn(2, 3)
##c_t_minus_1 = np.random.randn(2, 3)
            
##lstm = LSTM(W_h, W_x, b)
##x = np.array([[1, 2, 3], [1, 2, 3]])
##lstm.forward(x, h_t_minus_1, c_t_minus_1)
##lstm.backward(h_t_minus_1, c_t_minus_1)


##x = np.array([[[1, 2, 3], [1, 2, 3]]])
##rnn = Time_LSTM(W_h, W_x, b, 1)
##h, c = rnn.forward(x, h_t_minus_1, c_t_minus_1)
##dcnext, dhnext, dx = rnn.backward(1, 0)
##optimizer = SGD()
##optimizer.update(rnn.params, rnn.grads)




