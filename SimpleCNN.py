import sys, os
sys.path.append(os.pardir)
from mnist import load_mnist

import numpy as np
from collections import OrderedDict

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

    if y.size == x.size:
        y = y.argmax(axis=1)

    batch_size = x.shape[0]
    return -np.sum(np.log(x[np.arange(batch_size), y] + 1e-7)) / batch_size

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
        if self.t.size == self.y.size:
            dx = (self.y - self.t) / batch

        else:
            dx = self.y.copy()
            dx[np.arange(batch), self.t] -= 1
            dx = dx / batch

        return dx

class affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.x_shape = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x
        out = np.dot(x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        dx = dx.reshape(self.x_shape)
        return dx

def accuracy(x, y):
    if x.size == y.size:
        accuracy = np.sum(np.argmax(x, axis=1) == np.argmax(y, axis=1)) / float(x.shape[0])
    else:
        accuracy = np.sum(np.argmax(x, axis=1) == y) / float(x.shape[0])

    return accuracy



def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    n, c, h, w = input_data.shape
    out_h = (h + 2 * pad - filter_h) // stride + 1
    out_w = (w + 2 * pad - filter_w) // stride + 1

    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((n, c, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(n * out_h * out_w, -1)
    return col

def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0,3,4,5,1,2)
    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H+pad, pad:W+pad]

class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
        self.x = None
        self.col = None
        self.col_W = None
        self.dW = None
        self.db = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = (H + 2 * self.pad - FH) // self.stride + 1
        out_w = (W + 2 * self.pad - FW) // self.stride + 1

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T

        out = np.dot(col, col_W) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W
        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0,2,3,1).reshape(-1,FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dCol = np.dot(dout, self.col_W.T)
        dx = col2im(dCol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx


class MaxPooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = (H + 2 * self.pad - self.pool_h) // self.stride + 1
        out_w = (W + 2 * self.pad - self.pool_w) // self.stride + 1

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h * self.pool_w)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max
        return out

    def backward(self, dout):
        dout = dout.transpose(0,2,3,1)
        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)

        return dx

class AvgPooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        self.x = None
        self.avg_num = None

    def forward(self,x):
        N, C, H, W = x.shape
        out_h = (H + 2 * self.pad - self.pool_h) // self.stride + 1
        out_w = (W + 2 * self.pad - self.pool_w) // self.stride + 1

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad) #(n * out_h * out_w, c * pool_h * pool_w)
        col = col.reshape(-1, self.pool_h * self.pool_w) #(n * out_h * out_w * c , pool_h, pool_w)

        avg = np.average(col, axis=1)
        avg = avg.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)
        self.x = avg
        self.avg_num = col.shape[0]
        return avg

    def backward(self, dout):
        return (1 / self.avg_num) * dout

class SimpleCNN:
    def __init__(self, input_dim = (1, 28, 28), conv_param={'filter_n' : 30, 'filter_size' : 5, 'pad' : 0, 'stride' : 1},
                 hidden_size = 100, output_size = 10, weight_init_std = 0.01):
        filter_num = conv_param['filter_n']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_size = input_dim[1]
        conv_output_size = (input_size - filter_size + 2 * filter_pad) // filter_stride + 1
        pool_output_size = int(filter_num * (conv_output_size/2) * (conv_output_size/2))

        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num)
        self.params['W2'] = weight_init_std * np.random.randn(pool_output_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'], conv_param['stride'], conv_param['pad'])
        self.layers['Relu1'] = relu()
        self.layers['Pool1'] = MaxPooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Affine1'] = affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = relu()
        self.layers['Affine2'] = affine(self.params['W3'], self.params['b3'])
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
        dout = 1
        dout = self.softmax.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['W1'] = self.layers['Conv1'].dW
        grads['b1'] = self.layers['Conv1'].db
        grads['W2'] = self.layers['Affine1'].dW
        grads['b2'] = self.layers['Affine1'].db
        grads['W3'] = self.layers['Affine2'].dW
        grads['b3'] = self.layers['Affine2'].db

        return grads

    def train(self, data, label, batch, epoch, learning_rate):
        for i in range(epoch):
            batch_mask = np.random.choice(data.shape[0], batch)
            x = data[batch_mask]
            t = label[batch_mask]

            grads = self.gradient(x, t)
            for grad in grads:
                self.params[grad] -= learning_rate * grads[grad]

            train_acc = accuracy(self.predict(x), t)
            print('epoch {} accuracy : {}'.format(i, train_acc))

    def test(self, test_data, test_label):
        predict = self.predict(test_data)
        test_acc = accuracy(predict, test_label)
        return test_acc

(x_train, y_train), (x_test, y_test) = load_mnist(flatten=False)
model = SimpleCNN()
model.train(x_train, y_train, 100, 10000, 0.01)
test_acc = model.test(x_test, y_test)
print(test_acc)
