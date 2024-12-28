import numpy as np
import pandas as pd

import utils

mnist_train, mnist_test = utils.load_mnist()


class FullyConnectedNetwork:
    def __init__(self, sizes: list[int], path: str = None):
        self.num_layers = len(sizes)
        self.sizes = sizes[:]

        if path is None:
            # the biases are stored in a list of numpy arrays (column vectors):
            # the biases of the 2nd layer are stored in self.biases[1],
            # the biases of the 3rd layer are stored in self.biases[2], etc.
            # all layers but the input layer get biases
            self.biases = [None] + [np.random.randn(size, 1) for size in sizes[1:]]
            # initializing weights: list of numpy arrays (matrices)
            # self.weights[l][j][k] - weight from the k-th neuron in the l-th layer
            # to the j-th neuron in the (l+1)-th layer
            self.weights = [None] + [np.random.randn(sizes[i + 1], sizes[i]) for i in range(self.num_layers - 1)]

        else:
            self.biases = [None]
            self.weights = [None]

            for i in range(1, self.num_layers):
                biases = pd.read_csv(f"{path}/biases[{i}].csv", header=None).to_numpy()
                self.biases.append(biases)
                weights = pd.read_csv(f"{path}/weights[{i}].csv", header=None).to_numpy()
                self.weights.append(weights)

    def feedforward(self, input_vector):
        """
        Returns the output of the network, given a certain input
        :param input_vector: np.ndarray of shape (n, 1), where n = self.sizes[0] (size of input layer)
        :returns: np.ndarray of shape (m, 1), where m = self.sizes[-1] (size of output layer)
        """
        x = input_vector.copy()  # call copy constructor
        for i in range(1, self.num_layers):
            x = utils.sigmoid(np.dot(self.weights[i], x) + self.biases[i])
        return x

    def get_result(self, output):
        """
        Returns the digit corresponding to the output of the network
        :param output: np.ndarray of shape (m, 1), where m = self.sizes[-1] (size of output layer)
        :returns: int
        """
        result = 0
        for i in range(1, self.sizes[-1]):
            if output[i][0] > output[result][0]:
                result = i
        return result

    def test_network(self, testing_data=None):
        """
        Test the network
        :param testing_data: None or numpy.ndarray of shape (n, m), where n = total number of testing examples,
                                                                          m = self.sizes[0] + 1 (size of input layer + 1 for the label)
        :returns: None
        """
        if testing_data is None:
            testing_data = mnist_test.copy()
        total_correct = 0
        total = testing_data.shape[0]
        for i in range(total):
            input_vector = testing_data[i][1:] / 255  # normalize input, label is on column 0
            input_vector = input_vector[..., None]  # transforming 1D array into (n, 1) ndarray
            if self.get_result(self.feedforward(input_vector)) == testing_data[i][0]:
                total_correct += 1
        print(f"{total_correct}/{total}")

    def backprop(self, input_vector, y):
        """
        Backpropagation function.
        Returns the gradient of the cost function (MSE - Mean Squared Error) for a certain input
        :param input_vector: np.ndarray of shape (n, 1), where n = self.sizes[0] (size of input layer)
        :param y: np.ndarray of shape (m, 1), where m = self.sizes[-1] (size of output layer)
        :returns: gradient in terms of both weights and biases, w.r.t. the provided input
        """

        # forward propagation
        z = [None]
        a = [input_vector.copy()]
        for i in range(1, self.num_layers):
            z.append(np.dot(self.weights[i], a[-1]) + self.biases[i])
            a.append(utils.sigmoid(z[-1]))

        gradient_biases = [None] * self.num_layers
        gradient_weights = [None] * self.num_layers

        # backwards propagation
        error = utils.derivative_mse(y, a[-1]) * utils.derivative_sigmoid(z[-1])  # error in the output layer
        gradient_biases[-1] = error.copy()
        gradient_weights[-1] = np.dot(error, a[-2].T)
        for i in range(self.num_layers - 2, 0, -1):
            error = np.dot(self.weights[i + 1].T, error) * utils.derivative_sigmoid(z[i])  # error in the subsequent layer
            gradient_biases[i] = error.copy()
            gradient_weights[i] = np.dot(error, a[i - 1].T)

        return gradient_biases, gradient_weights

    def weights_biases_to_csv(self, path: str):
        for i in range(1, self.num_layers):
            biases = pd.DataFrame(self.biases[i])
            biases.to_csv(f"{path}/biases[{i}].csv", encoding="utf-8", index=False, header=False)
            weights = pd.DataFrame(self.weights[i])
            weights.to_csv(f"{path}/weights[{i}].csv", encoding="utf-8", index=False, header=False)

    def SGD(self, mini_batch_size, epochs, learning_rate, training_data=None):
        """
        Stochastic Gradient Descent
        :param mini_batch_size: int
        :param epochs: int
        :param learning_rate: float
        :param training_data: None or numpy.ndarray of shape (n, m), where n = total number of training examples, m = self.sizes[0] + 1 (size of input layer + 1 for the label)
        :returns: None
        """
        if training_data is None:
            training_data = mnist_train.copy()

        total_training_examples = training_data.shape[0]
        batches = total_training_examples // mini_batch_size

        for epoch in range(epochs):
            np.random.shuffle(training_data)

            for batch in range(batches):
                gradient_biases_sum = [None] + [np.zeros((size, 1)) for size in self.sizes[1:]]
                gradient_weights_sum = [None] + [np.zeros((self.sizes[i + 1], self.sizes[i])) for i in range(self.num_layers - 1)]

                for i in range(batch * mini_batch_size, (batch + 1) * mini_batch_size):
                    input_vector = training_data[i][1:] / 255  # position [i][0] is label
                    input_vector = input_vector[..., None]  # transforming 1D array into (n, 1) ndarray

                    y = training_data[i][0]
                    gradient_biases_current, gradient_weights_current = self.backprop(input_vector, utils.one_hot_encode(y, self.sizes[-1]))

                    for l in range(1, self.num_layers):
                        gradient_biases_sum[l] += gradient_biases_current[l]
                        gradient_weights_sum[l] += gradient_weights_current[l]

                for i in range(1, self.num_layers):
                    self.biases[i] -= learning_rate / mini_batch_size * gradient_biases_sum[i]
                    self.weights[i] -= learning_rate / mini_batch_size * gradient_weights_sum[i]

            # NOTE: if total_training_examples isn't divisible by mini_batch_size
            # range of inputs if total_training_examples % mini_batch_size != 0: range(batches * mini_batch_size, total_training_examples)
            # number of training inputs: total_training_examples % mini_batch_size
            if total_training_examples % mini_batch_size != 0:
                gradient_biases_sum = [None] + [np.zeros((size, 1)) for size in self.sizes[1:]]
                gradient_weights_sum = [None] + [np.zeros((self.sizes[i + 1], self.sizes[i])) for i in range(self.num_layers - 1)]

                for i in range(batches * mini_batch_size, total_training_examples):
                    input_vector = np.array(training_data[i][1:]) / 255  # position 0 is label
                    input_vector = input_vector[..., None]  # transforming 1D array into (n, 1) ndarray

                    y = training_data[i][0]
                    gradient_biases_current, gradient_weights_current = self.backprop(input_vector, utils.one_hot_encode(y, self.sizes[-1]))

                    for l in range(1, self.num_layers):
                        gradient_biases_sum[l] += gradient_biases_current[l]
                        gradient_weights_sum[l] += gradient_weights_current[l]

                for i in range(1, self.num_layers):
                    self.biases[i] -= (learning_rate / (total_training_examples % mini_batch_size)) * gradient_biases_sum[i]
                    self.weights[i] -= (learning_rate / (total_training_examples % mini_batch_size)) * gradient_weights_sum[i]

            # test the network in each epoch
            print(f"Epoch {epoch}: ", end="")
            self.test_network()


if __name__ == "__main__":
    # train the network
    np.random.seed(0)
    digit_recognizer = FullyConnectedNetwork([784, 64, 10])
    digit_recognizer.SGD(20, 10, 0.5)
    digit_recognizer.weights_biases_to_csv("../data/weights_biases/")  # write the weights and biases to files

    # test the network
    # digit_recognizer = Network([784, 32, 10], "../data/weights_biases/")
    # digit_recognizer.test_network()