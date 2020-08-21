import numpy as np
import RNN


def Softmax(x):
    denominator = np.sum(np.exp(x), axis = 1)
    denominator = denominator[:, np.newaxis]
    return np.exp(x) / np.exp(denominator)

def CrossEntropyLoss(y, y_hat):
    N = y.shape[0]
    log_probability = np.log(y_hat)
    total_loss = np.sum(y * log_probability)
    total_loss = total_loss * -1 
    return total_loss / N

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

class Affine:
    def __init__(self, input_size, hidden_size, batch_size = 1):
        self.W = np.random.randn(input_size, hidden_size)
        self.b =np.random.randn(batch_size, hidden_size)
        self.params = [self.W, self.b]
        self.grads = []
        self.x = None

    def forward(self, x):
        out = np.dot(x, self.W) + self.b
        self.x = x
        return out

    def backward(self, dout):
        self.grads = []
        dx = np.dot(dout, self.W.T)
        dW = np.dot(self.x.T, dout)
        db = dout.copy()

        self.grads.append(dW)
        self.grads.append(db)
        return dx

class SoftmaxwithLoss:
    def __init__(self):
        self.params = []
        self.y_hat = None
        self.y = None
    
    def forward(self, x, y):
        y_hat = Softmax(x)
        self.y_hat = y_hat
        self.y = y
        loss = CrossEntropyLoss(y, y_hat)
        return y_hat, loss

    def backward(self, dout=1):
        dimension = self.y_hat.shape[0]
        self.y_hat[self.y==1] -= 1
        dx = self.y_hat * dout
        dx /= dimension
        return dx

class RNNLM:
    def __init__(self, W_h, W_x, b, vocabuary_size, embedding_matrix):
        self.params = [W_h, W_x, b]
        self.grads = []
        hidden_size = W_h.shape[0]
        self.RNNLM = [Embedding_Layer(embedding_matrix),
                      RNN.RNN(W_h, W_x, b),
                      Affine(hidden_size, vocabuary_size)
                      ]
        self.loss_layer = SoftmaxwithLoss()
            
    def forward(self, input_word, h_t_minus_one, y):
        word_embbeded = self.RNNLM[0].forward(input_word)
        self.params.append(self.RNNLM[0].params)
        h_t = self.RNNLM[1].forward(word_embbeded, h_t_minus_one)
        affine = self.RNNLM[2].forward(h_t)
        self.params.append(self.RNNLM[2].params[0])
        self.params.append(self.RNNLM[2].params[1])
        y_hat, loss = self.loss_layer.forward(affine, y)
        predicted_word = np.argmax(y_hat, axis=1)
        return predicted_word, h_t, loss
    
    def backward(self, dhnext, dout = 1):
        W_h, W_x, b = self.params[0], self.params[1], self.params[2]
        self.grads = [np.zeros_like(W_h), np.zeros_like(W_x), np.zeros_like(b)]
        dout = self.loss_layer.backward(dout)
        dout = self.RNNLM[2].backward(dout)
        dhnext, dout = self.RNNLM[1].backward(dhnext + dout)
        dout = self.RNNLM[0].backward(dout)
        self.grads[0] += self.RNNLM[1].grads[0]
        self.grads[1] += self.RNNLM[1].grads[1]
        self.grads[2] += self.RNNLM[1].grads[2]
        self.grads.append(self.RNNLM[0].grads)
        self.grads.append(self.RNNLM[2].grads[0])
        self.grads.append(self.RNNLM[2].grads[1])
        return dhnext

class Time_RNNLM:
    def __init__(self, W_h, W_x, b, vocabuary_size, embedding_matrix, sequence_length):
        self.rnnlm = []
        self.predicted_word_list = []
        self.params = [W_h, W_x, b]
        self.grads = []
        for _ in range(sequence_length):
            self.rnnlm.append(RNNLM(W_h, W_x, b, vocabuary_size, embedding_matrix))

    def forward(self, x, y, h_t_minus_1):
        LOSS = 0
        predicted_word = x
        self.predicted_word_list = []
        for index, y_t in enumerate(y):
            y_t = np.array([y_t])
            predicted_word, h_t_minus_1, loss = self.rnnlm[index].forward(predicted_word, h_t_minus_1, y_t)
            self.params.append(self.rnnlm[index].params[3])
            self.params.append(self.rnnlm[index].params[4])
            self.params.append(self.rnnlm[index].params[5])
            LOSS += loss
            self.predicted_word_list.append(predicted_word)
        return self.predicted_word_list, LOSS

    def backward(self):
        dhnext = 0
        Wx, Wh, b = self.params[0], self.params[1], self.params[2]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        for reversed_rnnlm in reversed(self.rnnlm):
            dhnext = reversed_rnnlm.backward(dhnext, 1)
            self.grads[0] += reversed_rnnlm.grads[0]
            self.grads[1] += reversed_rnnlm.grads[1]
            self.grads[2] += reversed_rnnlm.grads[2]
        for rnnlm in self.rnnlm:
            self.grads.append(rnnlm.grads[3])
            self.grads.append(rnnlm.grads[4])
            self.grads.append(rnnlm.grads[5])

input_size = 2
hidden_size = 3
batch_size = 1

W_h = np.random.randn(hidden_size, hidden_size)
W_x = np.random.randn(input_size, hidden_size)
b = np.random.randn(batch_size, hidden_size)
h_t_minus_one = np.random.randn(1, 3)

a = np.array([[0, 1], [1, 2], [2, 3]])
y = np.array([[0, 0, 1], [0, 1, 0], [0, 0, 1]])
input_word = np.array([0])
rnn = Time_RNNLM(W_h, W_x, b, 3, a, 3)
predicted_word, loss = rnn.forward(input_word, y, h_t_minus_one)
rnn.backward()
print(len(rnn.grads))

print(len(rnn.params))





