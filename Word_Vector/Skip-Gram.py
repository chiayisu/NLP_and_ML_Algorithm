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

class SoftmaxwithLoss:
    def __init__(self):
        self.params = []
        self.y_hat = None
        self.y = None
    
    def forward(self, x, y):
        y_hat = Softmax(x)
        self.y = y
        self.y_hat = y_hat
        loss = CrossEntropyLoss(y, y_hat)
        return loss

    def backward(self, dout=1):
        dimension = self.y_hat.shape[0]
        self.y_hat[self.y==1] -= 1
        dx = self.y_hat * dout
        dx /= dimension
        return dx

class SGD:
    def __init__(self, lr = 0.01):
        self.lr = lr

    def update(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.lr * grads[i] 

class Skip_Gram:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.grads, self.params = [], []
        self.output_dim = output_dim
        self.Win = np.random.randn(input_dim, hidden_dim)
        self.Wout = np.random.randn(hidden_dim, output_dim)
        self.params.append(self.Wout)
        self.params.append(self.Win)
        self.loss_layer = SoftmaxwithLoss()
        self.center = None
        self.hidden_0 = None

    def predict(self, center):
        hidden_0 = np.dot(center, self.Win)
        hidden_1 = np.dot(hidden_0, self.Wout)
        output = Softmax(hidden_1)
        return output

    def forward(self, center, context):
        self.center = center
        hidden_0 = np.dot(center, self.Win)
        self.hidden_0 = hidden_0
        hidden_1 = np.dot(hidden_0, self.Wout)
        loss = self.loss_layer.forward(hidden_1, context)
        return loss

    def backward(self, dout = 1):
        self.grads = []
        dout = self.loss_layer.backward()
        dWout = np.dot(self.hidden_0.T, dout)
        dout = np.dot(dout, self.Wout.T)
        dWin = np.dot(self.center.T, dout)
        dout = np.dot(dout, self.Win.T)
        self.grads.append(dWout)
        self.grads.append(dWin)
        return dout

context = np.array([[0, 0, 1, 1, 0]])
center_word = np.array([[0, 1, 0, 0, 0]])
Word_Vector = Skip_Gram(5, 3, 5)
optimizer = SGD(0.0001)
loss_list = []
for epoch in range(30):
    loss = Word_Vector.forward(center_word, context)
    loss_list.append(loss)
    dout = Word_Vector.backward()
    optimizer.update(Word_Vector.params, Word_Vector.grads)
print(loss_list)
    
