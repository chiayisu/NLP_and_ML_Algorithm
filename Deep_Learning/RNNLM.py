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
    def __init__(self, input_size, hidden_size, batch_size, vocabuary_size, embedding_matrix):
        self.params = []
        self.grads = []
        self.RNNLM = [Embedding_Layer(embedding_matrix),
                      RNN.RNN(input_size, hidden_size, batch_size),
                      Affine(hidden_size, vocabuary_size)
                      ]
        self.loss_layer = SoftmaxwithLoss()
        for layer in reversed(self.RNNLM):
            self.params.append(layer.params)
    def forward(self, input_word, h_t_minus_one, y):
        word_embbeded = self.RNNLM[0].forward(input_word)
        h_t = self.RNNLM[1].forward(word_embbeded, h_t_minus_one)
        affine = self.RNNLM[2].forward(h_t)
        y_hat, loss = self.loss_layer.forward(affine, y)
        predicted_word = np.argmax(y_hat, axis=1)
        return predicted_word, h_t, loss
    
    def backward(self, dout = 1):
        self.grads = []
        dx = self.loss_layer.backward()
        dx = self.RNNLM[2].backward(dx)
        d_h_minus_1, dx = self.RNNLM[1].backward(dx)
        dx = self.RNNLM[0].backward(dx)
        for layer in reversed(self.RNNLM):
            self.grads.append(layer.grads)

class Time_RNNLM:
    def __init__(self, input_size, hidden_size, batch_size, vocabuary_size, embedding_matrix, sequence_length):
        self.rnnlm = []
        self.predicted_word_list = []
        for _ in range(sequence_length):
            self.rnnlm.append(RNNLM(input_size, hidden_size, batch_size, vocabuary_size, embedding_matrix))

    def forward(self, x, y, h_t_minus_1):
        LOSS = 0
        predicted_word = x
        self.predicted_word_list = []
        for index, y_t in enumerate(y):
            y_t = np.array([y_t])
            predicted_word, h_t_minus_1, loss = self.rnnlm[index].forward(predicted_word, h_t_minus_1, y_t)
            LOSS += loss
            self.predicted_word_list.append(predicted_word)
        return self.predicted_word_list, LOSS

    def backward(self):
        for reversed_rnnlm in reversed(self.rnnlm):
            reversed_rnnlm.backward(1)


h_t_minus_one = np.random.randn(1, 3)
a = np.array([[0, 1], [1, 2], [2, 3]])
y = np.array([[0, 0, 1], [0, 1, 0], [0, 0, 1]])
input_word = np.array([0])
rnn = Time_RNNLM(2, 3, 1, 3, a, 3)
predicted_word, loss = rnn.forward(input_word, y, h_t_minus_one)
rnn.backward()

