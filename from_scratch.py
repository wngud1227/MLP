import numpy as np

def sigmoid(xs):
    result = 1 / 1 + np.exp(-xs)
    return result


class affine:
    def __init__(self, W, b):   #W : (m,n), b : (m,1), x : (n, k)
        self.W, self.b, self.lr = W, b, 0.025
        self.cache = None
    def forward(self, x):
        h = np.dot(self.W, x) + self.b
        y = sigmoid(h)          #y : (m, k)
        self.cache = (x, y)
        return y
    def backward(self, dout):
        (x, y) = self.cache
        # dout *= np.multiply(y, (1 - y))  #dout : (m , k)
        dx = np.dot(self.W.T, dout) #dx : (m, k)
        dW = np.dot(dout, x.T)      #dW : (m, n)

        self.b -= np.sum(dout, axis=0) * self.lr
        self.W -= dW * self.lr

        return dx

class softmax_with_loss:
    def __init__(self):
        self.cache = None

    def forward(self, x, t):
        y = np.exp(x) / np.sum(np.exp(x))   #y : (m, k)
        loss = np.sum(np.log(y[:, t.argmax()] + 1e-07))
        self.cache = (y, t)
        return loss

    def backward(self):
        (y, t) = self.cache
        dout = y - t
        return dout

class MLP:
    def __init__(self, input, label, batch_size):
        self.input = input
        self.t = label
        self.batch_size = batch_size
        self.cache = None
        self.W = None
        self.b = None

    def weight_initialize(self, hidden_size): #ex) hidden_size = [300, 100, 10]
        W = []
        b = []
        out = self.t.shape[0]
        len = self.input.shape[0]
        for i in range(len(hidden_size)):
            weight = np.zeros((len, hidden_size[i]), dtype='f')
            bias = np.zeros((hidden_size[i]), dtype='f')
            W.append(weight)
            b.append(bias)
            len = weight.shape[1]
        weight = np.zeros((len, out))
        W.append(weight)

        self.W, self.b = W, b
        return None

    def forward(self, x, t):
        W = self.W.copy()
        b = self.b.copy()
        cache = []
        for i in range(len(b)):
            y = sigmoid(np.matmul(W[i], x) + b)
            cache.append((x, y))
            x = y
        h = np.matmul(W[-1], x)
        y = np.exp(h) / np.sum(np.exp(h), axis=0)
        cache.append((x, y, t))
        self.cache = cache[::-1].copy()

        loss = np.sum(np.log(y[:, t.argmax()] + 1e-07))
        return loss

    def backward(self, lr):
        cache = self.cache.copy()
        W, b = self.W.copy(), self.b.copy()
        (x, y, t) = cache[0]
        dout = y - t
        dW = np.dot(dout, x.T)
        W[-1] -= dW * lr

        for i in range(len(cache) - 2, 0, -1):
            (x, y) = cache[i]
            dout = dx * y * (1 - y)  #dout : (m , k)
            dx = np.dot(W[i].T, dout)  # dx : (m, k)
            dW = np.dot(dout, x.T)  # dW : (m, n)

            b[i] -= np.sum(dout, axis=0) * lr
            W[i] -= dW * lr

        return None

def train(model, epoch, input, label, batch_size, lr):
    model = model(input, label, batch_size)
    for i in range(epoch):
        #select mini batch
        x_batch, t_batch = input[:], label[:]
        loss = model.forward(x_batch, t_batch)
        print(loss)
        model.backward(lr)

    return model


a = np.ones((3, 4))
print(a * a)
