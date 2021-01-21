import numpy as np
from tensorflow.keras.datasets import mnist

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=7000)]
        )

    except RuntimeError as e:
        print(e)

def one_hot(x):
    y = []
    for i in x:
        if i == 0:
          y.append([1,0,0,0,0,0,0,0,0,0])
        elif i == 1:
          y.append([0,1,0,0,0,0,0,0,0,0])
        elif i == 2:
          y.append([0,0,1,0,0,0,0,0,0,0])
        elif i == 3:
          y.append([0,0,0,1,0,0,0,0,0,0])
        elif i == 4:
          y.append([0,0,0,0,1,0,0,0,0,0])
        elif i == 5:
          y.append([0,0,0,0,0,1,0,0,0,0])
        elif i == 6:
          y.append([0,0,0,0,0,0,1,0,0,0])
        elif i == 7:
          y.append([0,0,0,0,0,0,0,1,0,0])
        elif i == 8:
          y.append([0,0,0,0,0,0,0,0,1,0])
        elif i == 9:
          y.append([0,0,0,0,0,0,0,0,0,1])

    y = np.array(y)
    return y

#matrix

def matrix_sum(x, y):
    result = []
    for i in range(y.size):
        result.append(x[i] + y[0][i])
    result = np.array(result)
    result = result.reshape(1, x.size)
    return result


def matrix_mul(x, y):
    result = []
    for i in x:
        for j in y:
            result.append(i * j)
    result = np.array(result)
    result = result.reshape(x.size, y.size)
    return result

#activation function

class sigmoid:
    def function(x):
        y = []
        for i in x:
            y.append(1 / (1 + np.exp(-i)))
        y = np.array(y)
        return y

    def derivative(x):
        y = []
        for i in x:
            y.append(i * (1 - i))
        y = np.array(y)
        return y

class relu:
    def function(x):
        y = []
        for i in x:
            y.append(np.maximum(0, i))
        y = np.array(y)
        return y

    def derivative(x):
        y = []
        for i in x:
            y.append((i > 0).astype(i.dtype))
        y = np.array(y)
        return y

class tanh:
    def function(x):
        y = []
        for i in x:
            y.append(np.tanh(i))
        y = np.array(y)
        return y

    def derivative(x):
        y = []
        for i in x:
            y.append(1 - (i ** 2))
        y = np.array(y)
        return y

#softmax

class softmax:
    def function(x):
        return np.exp(x) / np.exp(x).sum()

    def derivative(x):
        return np.matmul(x.T, x)


#loss

class cross_entropy:

    def function(x, y):
        z = x * np.log(y)
        return -(z.sum())

    def derivative(x, y):
        return -(np.log(y))

class MSE:
    def function(x, y):
        return 0.5 * np.sum((y - x) ** 2)
    def derivative(x, y):
        return x - y


#accuracy
def accuracy(predict, label, result):
    temp = list(predict)
    index = temp.index(max(temp))
    if label[index] == 1:
        result += 1
    return result


class MLP:
    def __init__(self, data, label, activation, loss, epoch, lr, layer):
        self.data = data.reshape(60000, 784)
        self.size = data.shape
        self.epoch = epoch
        self.label = label
        self.lr = lr
        self.layer = 0
        self.max_layer = layer

        try:
            if activation == 'relu':
                self.activation = relu.function
                self.derivative = relu.derivative
            elif activation == 'sigmoid':
                self.activation = sigmoid.function
                self.derivative = sigmoid.derivative
            elif activation == 'tanh':
                self.activation = tanh.function
                self.derivative = tanh.derivative
        except:
            print('Activation function error!')

        try:
            if loss == 'cross_entropy':
                self.loss = cross_entropy.function
                self.loss_derivative = cross_entropy.derivative
            elif loss == 'MSE':
                self.loss = MSE.function
                self.loss_derivative = MSE.derivative
        except:
            print("Loss function error!")

    def weight_init(self, n_b, n_f):
        return np.random.rand(n_b, n_f)

    def feedforward(self, weight, data, bias):
        hidden_state = np.matmul(data, weight)
        hidden_state = matrix_sum(hidden_state, bias)
        self.layer += 1
        return hidden_state

    def backpropagation(self, weight, data, alpha, label):
        try:
            if self.layer == self.max_layer:
                loss = self.loss_derivative(data, label)
                derivative = softmax.derivative(data)
                self.layer -= 1
                return np.matmul(loss, derivative)

            elif self.layer > 0:
                temp = np.matmul(alpha, weight.T)
                derivative = self.derivative(data)
                self.layer -= 1
                return np.multiply(temp, derivative)

        except:
            print('backpropagation error!')


#train
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


with tf.device('/device:GPU:0'):
    y_one = one_hot(y_train)
    MLP = MLP(x_train, y_one, 'sigmoid', 'MSE', 10, 0.001, 3)

    w1 = MLP.weight_init(784, 100)
    w2 = MLP.weight_init(100, 30)
    w3 = MLP.weight_init(30, 10)

    b1 = np.zeros((1, 100))
    b2 = np.zeros((1, 30))
    b3 = np.zeros((1, 10))

    for j in range(MLP.epoch):
        for i in range(len(MLP.label)):
            h1 = MLP.feedforward(w1, MLP.data[i], b1)
            h1 = MLP.activation(h1)

            h2 = MLP.feedforward(w2, h1[0], b2)
            h2 = MLP.activation(h2)

            h3 = MLP.feedforward(w3, h2[0], b3)
            predict = softmax.function(h3)

            back3 = MLP.backpropagation(weight=None, data=predict, alpha=None, label=MLP.label[i])
            back2 = MLP.backpropagation(weight=w3, data=h2, alpha=back3, label=MLP.label[i])
            back1 = MLP.backpropagation(weight=w2, data=h1, alpha=back2, label=MLP.label[i])

            w3 -= MLP.lr * np.matmul(h2.T, back3)
            w2 -= MLP.lr * np.matmul(h1.T, back2)
            w1 -= MLP.lr * matrix_mul(MLP.data[i].T, back1)

            b3 -= MLP.lr * back3
            b2 -= MLP.lr * back2
            b1 -= MLP.lr * back1
        print("{} epoch finished".format(j + 1))

#test
    test_label = one_hot(y_test)
    test_accuracy = 0
    test_data = x_test.reshape(10000, 784)
    for i in range(len(test_data)):
        t1 = MLP.feedforward(w1, test_data[i], b1)
        t1 = MLP.activation(t1)

        t2 = MLP.feedforward(w2, t1[0], b2)
        t2 = MLP.activation(t2)

        t3 = MLP.feedforward(w3, t2[0], b3)
        test_predict = softmax.function(t3)

        test_accuracy = accuracy(test_predict, test_label[i], test_accuracy)

    test_accuracy = test_accuracy / len(test_data)
    print(test_accuracy)
