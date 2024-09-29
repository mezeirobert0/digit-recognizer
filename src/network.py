import numpy as np
import pandas as pd


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def derivative_sigmoid(z):
    s = sigmoid(z)
    return s * (1 - s)


class Network():
    def __init__(self, sizes: list[int]):
        self.num_layers = len(sizes)
        self.sizes = sizes[:]

        # the biases are stored in a list of numpy arrays (column vectors):
        # the biases of the 2nd layer are stored in self.biases[1],
        # the biases of the 3rd layer are stored in self.biases[2], etc.
        # all layers but the input layer get biases
        self.biases = [None] + [np.random.randn(size, 1) for size in sizes[1:]]
        # initializing weights: list of numpy arrays (matrices)
        # self.weights[l][j][k] - weight from the k-th neuron in the l-th layer
        # to the j-th neuron in the (l+1)-th layer
        self.weights = [None] + [np.random.randn(sizes[i + 1], sizes[i])
                                 for i in range(self.num_layers - 1)]
        # unactivated neurons
        self.z = [np.zeros((sizes[i], 1), dtype=int) for i in range(self.num_layers)]

    def feedforward(self, input):
        """
        Returns the output of the network, given a certain input

        :param input: np.ndarray of shape (n ,1), where n = self.sizes[0] (size of input layer)
        :returns: np.ndarray of shape (m ,1), where m = self.sizes[-1] (size of output layer)
        """
        x = np.array(input)  # call copy constructor
        for i in range(1, self.num_layers):
            x = sigmoid(np.dot(self.weights[i], x) + self.biases[i])
        return x

    def get_result(self, output):
        """
        Returns the digit corresponding to the output of the network

        :param output: np.ndarray of shape (m ,1), where m = self.sizes[-1] (size of output layer) (components should add up to 1)
        :returns: int
        """
        result = 0
        for i in range(1, self.sizes[-1]):
            if output[i][0] > output[result][0]:
                result = i
        return result

    def get_expected_output(self, expected_result):
        """
        Returns the vector corresponding to the expected output of the network

        :param expected_result: int
        :returns: np.ndarray of shape (m, 1), where m = self.sizes[-1] (size of output layer)
        """
        expected_output = np.zeros((self.sizes[-1], 1), dtype=int)
        expected_output[expected_result, 0] = 1
        return expected_output

    def test_network(self, testing_data=None):
        if testing_data is None:
            testing_data = pd.read_csv('../datasets/mnist_test.csv')
            testing_data = testing_data.to_numpy()
        total_correct = 0
        total = testing_data.shape[0]
        for i in range(total):
            input_vector = testing_data[i][1:]  # label is on column 0
            # transforming 1D array into (n, 1) array
            input_vector = input_vector[..., None]
            if self.get_result(self.feedforward(input_vector)) == testing_data[i][0]:
                total += 1
        print(f'{total_correct}/{total}')

    def backprop(self, input, y):
        """
        Backpropagation function.
        Returns the gradient of the cost function (MSE - Mean Squared Error) for a certain input

        :param input: np.ndarray of shape (n, 1), where n = self.sizes[0] (size of input layer)
        :param y: np.ndarray of shape (m, 1), where m = self.sizes[-1] (size of output layer)
        :returns: None (yet)
        """
        # forward propagation - populating the neurons with their unactivated output
        z = [np.zeros((self.sizes[i], 1), dtype=int) for i in range(self.num_layers)]
        z[0] = np.array(input)  # call copy constructor
        for i in range(1, self.num_layers):
            a = sigmoid(z[i - 1])  # activate previous layer
            z[i] = sigmoid(np.dot(self.weights[i], a) + self.biases[i])

        # populating the errors in each layer (except the input one)
        errors = [np.zeros((self.sizes[i], 1), dtype=int) for i in range(self.num_layers)]
        errors[-1] = (sigmoid(z[-1]) - y) * derivative_sigmoid(z[-1])  # error in the output layer
        for i in range(self.num_layers - 2, 1, -1):
            errors[i] = np.dot(self.weights[i + 1].T, errors[i + 1]) * derivative_sigmoid(z[i])  # errors in the subsequent layers

        # TODO: calculate gradient components w.r.t. the weights and biases in each layer
        # then return these (somehow)

    def SDG(self, mini_batch_size, epochs, learning_rate, training_data=None):
        """Stochastic Gradient Descent"""
        if training_data is None:
            training_data = pd.read_csv('../datasets/mnist_train.csv')
            training_data = training_data.to_numpy()
        for epoch in range(epochs):
            np.random.shuffle(training_data)
            total_training_examples = training_data.shape[0]
            batches = total_training_examples // mini_batch_size
            for batch in range(batches):
                # NOTE: range of inputs: range(batch * mini_batch_size, (batch + 1) * mini_batch_size)
                # TODO: continue writinig SDG function using backprop function (for each input in the range above)
                pass


digit_recognizer = Network([784, 128, 10])
training_data = pd.read_csv('../datasets/mnist_train.csv')
# the datasets have the following format:
# label 1x1 1x2 ... 28x27 28x28
training_data = training_data.to_numpy()
digit_recognizer.SDG(training_data, 20, 10, 10)
