from LSTM import Time_LSTM, LSTM
import numpy as np

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

class Encoder:
    def __init__(self,W_h, W_x, b, sequence_size, embedding_matrix):
        self.lstm = Time_LSTM(W_h, W_x, b, sequence_size)
        self.embedding_layer = []
        self.word_embbed = []
        self.params = [W_x, W_h, b]
        self.grads = []
        self.sequence_size = sequence_size
        for _ in range(sequence_size):
            self.embedding_layer.append(Embedding_Layer(embedding_matrix))

    def forward(self, x, h_t_minus_1, c_minus_1):
        for index in range(self.sequence_size):
            embedding = self.embedding_layer[index].forward(x[index])
            self.word_embbed.append(embedding)
            self.params.append(self.embedding_layer[index])
        self.word_embbed = np.array(self.word_embbed)
        h_t_minus_1, c_minus_1 = self.lstm.forward(self.word_embbed, h_t_minus_1, c_minus_1)
        return h_t_minus_1[len(h_t_minus_1) - 1]
    
    def backward(self, dhnext, dcnext):
        dh_list, dc_list, dx_list = self.lstm.backward(dhnext, dcnext)
        self.grads = self.lstm.grads.copy()
        for index, embbed in enumerate(reversed(self.embedding_layer)):
            embbed.backward(np.array(dx_list[index]))
            self.grads.append(self.embedding_layer[index].grads)



class Decoder:
    def __init__(self, W_h, W_x, b, sequence_size, vocab_size, embedding_matrix):
        self.params = [W_x, W_h, b]
        self.grads = []
        self.embedding_layer = []
        self.lstm = []
        self.affine_layer = []
        self.loss_layer = []
        self.sequence_size = sequence_size
        hidden_size = W_h.shape[0]
        for _ in range(sequence_size):
            self.embedding_layer.append(Embedding_Layer(embedding_matrix))
            self.lstm.append(LSTM(W_h, W_x, b))
            self.affine_layer.append(Affine(hidden_size, vocab_size))
            self.loss_layer.append(SoftmaxwithLoss())
            
    def forward(self, x, h_t_minus_1, c_minus_1, y):
        LOSS = 0
        y_list = []
        for index in range(self.sequence_size):
            embedding = self.embedding_layer[index].forward(x)
            self.params.append(self.embedding_layer[index].params)
            h_t_minus_1, c_minus_1 = self.lstm[index].forward(embedding, h_t_minus_1, c_minus_1)
            affine = self.affine_layer[index].forward(h_t_minus_1)
            self.params.append(self.affine_layer[index].params[0])
            self.params.append(self.affine_layer[index].params[1])
            y_hat, loss = self.loss_layer[index].forward(affine, y[index])
            LOSS += loss
            x = np.array(np.argmax(y_hat, axis = 1))
            y_list.append(x)
        return y_list, LOSS
    
    def backward(self, dhnext, dcnext, dout = 1):
        Wx, Wh, b = self.params[0], self.params[1], self.params[2]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        dh_list = []
        dc_list = []
        last_index = self.sequence_size - 1
        for index in range(last_index, -1, -1):
            dout = self.loss_layer[index].backward(dout)
            dout = self.affine_layer[index].backward(dout)
            dcnext, dhnext, dx = self.lstm[index].backward(dhnext + dout, dcnext)
            self.grads[0] += self.lstm[index].grads[0]
            self.grads[1] += self.lstm[index].grads[1]
            self.grads[2] += self.lstm[index].grads[2]
            self.embedding_layer[index].backward(dx)
        for index in range(self.sequence_size):
            self.grads.append(self.embedding_layer[index].grads)
            self.grads.append(self.affine_layer[index].grads[0])
            self.grads.append(self.affine_layer[index].grads[1])
        return dhnext

class PeekyDecoder:
    def __init__(self, W_h, W_x, b, sequence_size, vocab_size, embedding_matrix):
        self.params = [W_x, W_h, b]
        self.grads = []
        self.embedding_layer = []
        self.lstm = []
        self.affine_layer = []
        self.loss_layer = []
        self.sequence_size = sequence_size
        hidden_size = W_h.shape[0] * 2
        self.embbed_dim = embedding_matrix.shape[1]
        for _ in range(sequence_size):
            self.embedding_layer.append(Embedding_Layer(embedding_matrix))
            self.lstm.append(LSTM(W_h, W_x, b))
            self.affine_layer.append(Affine(hidden_size, vocab_size))
            self.loss_layer.append(SoftmaxwithLoss())
            
    def forward(self, x, h_t_minus_1, c_minus_1, y):
        LOSS = 0
        y_list = []
        h_encoder = h_t_minus_1.copy()
        for index in range(self.sequence_size):
            embedding = self.embedding_layer[index].forward(x)
            self.params.append(self.embedding_layer[index].params)
            h_t_minus_1, c_minus_1 = self.lstm[index].forward(np.concatenate((embedding, h_t_minus_1), axis=1), h_t_minus_1, c_minus_1)
            affine_input = np.concatenate((h_encoder, h_t_minus_1), axis=1)
            affine = self.affine_layer[index].forward(affine_input)
            self.params.append(self.affine_layer[index].params[0])
            self.params.append(self.affine_layer[index].params[1])
            y_hat, loss = self.loss_layer[index].forward(affine, y[index])
            LOSS += loss
            x = np.array(np.argmax(y_hat, axis = 1))
            y_list.append(x)
        return y_list, LOSS
    
    def backward(self, dhnext, dcnext, dout = 1):
        Wx, Wh, b = self.params[0], self.params[1], self.params[2]
        hidden_dim = Wh.shape[0]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        dh_list = []
        dc_list = []
        last_index = self.sequence_size - 1
        for index in range(last_index, -1, -1):
            dout = self.loss_layer[index].backward(dout)
            dout = self.affine_layer[index].backward(dout)
            dout = dout[:,:hidden_dim]
            dcnext, dhnext, dx = self.lstm[index].backward(dhnext + dout, dcnext)
            self.grads[0] += self.lstm[index].grads[0]
            self.grads[1] += self.lstm[index].grads[1]
            self.grads[2] += self.lstm[index].grads[2]
            self.embedding_layer[index].backward(dx[:,:self.embbed_dim])
        for index in range(self.sequence_size):
            self.grads.append(self.embedding_layer[index].grads)
            self.grads.append(self.affine_layer[index].grads[0])
            self.grads.append(self.affine_layer[index].grads[1])
        return dhnext
        
class Seq2Seq:
    def __init__(self, EN_W_h, EN_W_x, EN_b, DE_W_h, DE_W_x, DE_b, EN_sequence_size, DE_sequence_Size, vocab_size, encoder_embedding_matrix, decoder_embedding_matrix):
        self.params = []
        self.grads = []
        self.encoder = Encoder(EN_W_h, EN_W_x, EN_b, EN_sequence_size, encoder_embedding_matrix)
        self.decoder = Decoder(DE_W_h, DE_W_x, DE_b, DE_sequence_Size, vocab_size, decoder_embedding_matrix)

    def forward(self, x_encoder, x_decoder, h_t_minus_1, c_minus_1, y):
        encoder_h_t_last = self.encoder.forward(x_encoder, h_t_minus_1, c_minus_1)
        y_list, loss = self.decoder.forward(x_decoder, encoder_h_t_last, c_minus_1, y)
        return y_list, loss

    def backward(self, DE_dhnext = 0, DE_dcnext = 0, EN_dcnext = 0, dout = 1):
        self.grads = []
        dh_decoder = self.decoder.backward(DE_dhnext, DE_dcnext, dout)
        self.encoder.backward(dh_decoder, EN_dcnext)
        for param in self.decoder.params:
            self.params.append(param)
        for param in self.encoder.params:
            self.params.append(param)
        for grads in self.decoder.grads:
            self.grads.append(grads)
        for grads in self.encoder.grads:
            self.grads.append(grads)
    
input_size = 2
hidden_size = 3
batch_size = 2

W_h = np.random.randn(hidden_size, 4 * hidden_size)
W_x = np.random.randn(input_size, 4 * hidden_size)
b = np.random.randn(batch_size, 4 * hidden_size)

h_t_minus_one = np.random.randn(2, 3)
c_t_minus_one = np.random.randn(2, 3)

a = np.array([[0.5, 1.1], [1.3, 2.2], [2.1, 3.3]])
y = np.array([[[0, 0, 1], [0, 1, 0]], [[0, 0, 1], [0, 1, 0]]])

## Decoder Test
input_word = np.array([0, 0])
##rnn = Decoder(W_h, W_x, b, 2, 3, a)
##rnn.forward(input_word, h_t_minus_one, c_t_minus_one, y)
##rnn.backward(0, 0)

## Encoder Test
input_word_en = np.array([[0, 0], [0, 1], [2, 0]])
##rnn = Encoder(W_h, W_x, b, 3, a)
##rnn.forward(input_word, h_t_minus_one, c_t_minus_one)
##print(len(rnn.params))
##rnn.backward(1, 1)




peeky_W_h = np.random.randn(hidden_size, 4 * hidden_size)
peeky_W_x = np.random.randn(input_size + hidden_size, 4 * hidden_size)
peeky_b = np.random.randn(batch_size, 4 * hidden_size)

h_t_minus_one = np.random.randn(2, 3)
c_t_minus_one = np.random.randn(2, 3)
## PeekyDecoder Test
input_word = np.array([0, 0])
rnn = PeekyDecoder(peeky_W_h, peeky_W_x, b, 2, 3, a)
rnn.forward(input_word, h_t_minus_one, c_t_minus_one, y)
rnn.backward(0, 0)




## seq2seq
##seq2seq = Seq2Seq(W_h, W_x, b, W_h, W_x, b, 2, 2, 3, a, a)
##y_list, loss = seq2seq.forward(input_word_en, input_word, h_t_minus_one, c_t_minus_one, y)
##seq2seq.backward()








