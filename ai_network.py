import random
import matplotlib.pyplot as plt
import numpy as np

class Layer:
    def __init__(self,size,input_size=None):
        if input_size == None:
            self.input_size = size
        self.input_size = input_size
        self.size = size
        self.weights = np.random.rand(input_size,size)
        self.bias_vec = np.random.rand(1, size) - 0.5

    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        # dBias = output_error

        # update parameters
        self.weights -= learning_rate * weights_error
        self.bias_vec -= learning_rate * output_error
        return input_error

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias_vec
        return self.output

def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2));

def mse_prime(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size;


class ActivationLayer(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    # return the activation input
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return (self.output)

    # return input_error = dE/dX for a given output_error=dE/dY
    def backward_propagation(self, output_error, learning_rate):
        return (self.activation_prime(self.input) * output_error)

class Netowrk:
    def __init__(self,learnig_rate = 0.1):
        self.layers = []
        self.input_size = 0

    def set_input(self,size):
        self.input_size = size

    def add_layer(self,layer):
        self.layers.append(layer)

    def predict(self,inputs):
        samples = len(inputs)
        result = []

        # run network over all samples
        for i in range(samples):
            # forward propagation
            output = inputs[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)

        return result

    def get_last_layer_size(self):
        if len(self.layers)==0:
            return self.input_size
        return self.layers[-1].size


    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        # dBias = output_error

        # update parameters
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error

    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    def fit(self, x_train, y_train, epochs, learning_rate):
        # sample dimension first
        samples = len(x_train)

        # training loop
        for i in range(epochs):
            err = 0
            for j in range(samples):
                # forward propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # compute loss (for display purpose only)
                err += self.loss(y_train[j], output)

                # backward propagation
                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            # calculate average error on all samples
            err /= samples
            print('epoch %d/%d   error=%f' % (i + 1, epochs, err))



def make_dataset():
    x_data = []
    y_data = []
    for i in range(100):
        x_data.append([random.randint(0,10),random.randint(0,10)])
        y_data.append([0,1])
        x_data.append([random.randint(10, 15), random.randint(10, 15)])
        y_data.append([1, 0])

    for index,i in enumerate(x_data):
        if y_data[index] == [0,1]:
            plt.plot(i[0],i[1],"ro")
        else:
            plt.plot(i[0], i[1],"bo")
    return x_data,y_data
if __name__ == "__main__":
    x_data,y_data = make_dataset()

    netowrk = Netowrk()
    netowrk.set_input(2)
    netowrk.add_layer(ActivationLayer(mse,mse_prime))
    for i in range(1):
        size = netowrk.get_last_layer_size()
        netowrk.add_layer(Layer(3,size))
        netowrk.add_layer(ActivationLayer(mse, mse_prime))
    size = netowrk.get_last_layer_size()
    netowrk.add_layer(Layer(2,size))
    print("done")

    x_train = np.array(x_data)
    y_train = np.array(y_data)

    netowrk.use(mse, mse_prime)

    netowrk.fit(x_train, y_train, epochs=50, learning_rate=0.01)
    print(netowrk.predict([[1,0]]))
