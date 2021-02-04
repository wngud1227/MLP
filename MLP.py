import sys, os
sys.path.append(os.pardir)
from mnist import load_mnist


import numpy as np
from collections import OrderedDict

#initialization
def Random_initialization(n_in, n_out):
    return np.random.randn(n_in, n_out)

def Xavier_initialization(n_in, n_out):
    weight = np.random.randn(n_in, n_out)
    var = np.sqrt(n_in)
    return weight / var

def He_initialization(n_in, n_out):
    weight = np.random.randn(n_in, n_out)
    var = np.sqrt(n_in / 2)
    return weight / var

#activation function

class sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out

        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx

class tanh:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = np.tanh(x)
        self.out = out

        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * (1.0 + self.out)
        return dx

class relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx

#softmax

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x -= np.max(x)
    return np.exp(x) / np.exp(x).sum()


#loss

def cross_entropy(x, y):
    if x.ndim == 1:
        x = x.reshape(1, x.size)
        y = y.reshape(1, y.size)
    batch_size = x.shape[0]
    return -np.sum(y * np.log(x + 1e-7)) / batch_size

def MSE(x, y):
    return 0.5 * np.sum((x - y) ** 2)

class Softmax:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch = self.t.shape[0]
        dx = (self.y - self.t) / batch

        return dx

class affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        return dx

def accuracy(x, y):
    accuracy = np.sum(np.argmax(x, axis=1) == np.argmax(y, axis=1)) / float(x.shape[0])
    return accuracy

class MLP:
    def __init__(self, data, label, weight_init, activation):
        self.data = data
        self.label = label
        self.input_size = data.shape[1]
        self.output_size = label.shape[1]
        self.layer = 3
        self.params = {}
        self.layers = OrderedDict()

        if weight_init == 'Xavier' or 'xavier':
            weight = Xavier_initialization

        elif weight_init == 'He' or 'he':
            weight = He_initialization

        else:
            weight = Random_initialization

        try:
            if activation == 'relu' or 'Relu':
                self.activation = relu
            elif activation == 'sigmoid' or 'Sigmoid':
                self.activation = sigmoid
            elif activation == 'tanh' or 'Tanh':
                self.activation = tanh

        except:
            print('Activation function error!')

        self.params['W1'] = weight(self.input_size, 400)
        self.params['b1'] = np.zeros(400)
        self.params['W2'] = weight(400, 300)
        self.params['b2'] = np.zeros(300)
        self.params['W3'] = weight(300, 10)
        self.params['b3'] = np.zeros(10)

        self.layers['A1'] = affine(self.params['W1'], self.params['b1'])
        self.layers['F1'] = self.activation()
        self.layers['A2'] = affine(self.params['W2'], self.params['b2'])
        self.layers['F2'] = self.activation()
        self.layers['A3'] = affine(self.params['W3'], self.params['b3'])
        self.softmax = Softmax()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.softmax.forward(y, t)

    def gradient(self, x, t):
        self.loss(x, t)
        temp = self.softmax.backward(1)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            temp = layer.backward(temp)

        grads = {}
        grads['W1'] = self.layers['A1'].dW
        grads['b1'] = self.layers['A1'].db
        grads['W2'] = self.layers['A2'].dW
        grads['b2'] = self.layers['A2'].db
        grads['W3'] = self.layers['A3'].dW
        grads['b3'] = self.layers['A3'].db

        return grads

    def train(self, batch, epoch, learning_rate):
        for i in range(epoch):
            batch_mask = np.random.choice(self.data.shape[0], batch)
            x = self.data[batch_mask]
            t = self.label[batch_mask]

            grads = self.gradient(x, t)
            for grad in grads:
                self.params[grad] -= learning_rate * grads[grad]

            train_acc = accuracy(self.predict(x), t)
            print('epoch {} accuracy : {}'.format(i, train_acc))


    def test(self, test_set, test_label):
        predict = self.predict(test_set)
        test_acc = accuracy(predict, test_label)
        return test_acc

(x_train, y_train), (x_test, y_test) = load_mnist(flatten=True, normalize=True, one_hot_label=True)
model = MLP(data=x_train, label=y_train, activation='relu', weight_init='he')
model.train(batch=100, epoch=10000, learning_rate=0.0005)
test_acc = model.test(x_test, y_test)
print(test_acc)
