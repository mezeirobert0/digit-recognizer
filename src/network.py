import numpy as np
import pandas as pd


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def derivative_sigmoid(z):
    s = sigmoid(z)
    return s * (1.0 - s)


mnist_train_df = pd.read_csv('../datasets/mnist_train.csv')
mnist_test_df = pd.read_csv('../datasets/mnist_test.csv')


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
        self.weights = [None] + [np.random.randn(sizes[i + 1], sizes[i]) for i in range(self.num_layers - 1)]

    def feedforward(self, input):
        """
        Returns the output of the network, given a certain input

        :param input: np.ndarray of shape (n, 1), where n = self.sizes[0] (size of input layer)
        :returns: np.ndarray of shape (m, 1), where m = self.sizes[-1] (size of output layer)
        """
        x = np.array(input)  # call copy constructor
        for i in range(1, self.num_layers):
            x = sigmoid(np.dot(self.weights[i], x) + self.biases[i])
        return x

    def get_result(self, output):
        """
        Returns the digit corresponding to the output of the network

        :param output: np.ndarray of shape (m, 1), where m = self.sizes[-1] (size of output layer) (real components, should add up to 1)
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
        expected_output = np.zeros((self.sizes[-1], 1))
        expected_output[expected_result][0] = 1
        return expected_output

    def test_network(self, testing_data=None):
        """
        Test the network

        :param testing_data: None or numpy.ndarray of shape (n, m), where n = total number of testing examples,
                                                                          m = self.sizes[0] + 1 (size of input layer + 1 for the label)
        :returns: None (stdout)
        """
        if testing_data is None:
            testing_data = mnist_test_df
            testing_data = testing_data.to_numpy()
        total_correct = 0
        total = testing_data.shape[0]
        for i in range(total):
            input_vector = testing_data[i][1:]  # label is on column 0
            input_vector = input_vector[..., None]  # transforming 1D array into (n, 1) ndarray
            if self.get_result(self.feedforward(input_vector)) == testing_data[i][0]:
                total_correct += 1
        print(f'{total_correct}/{total}')

    def backprop(self, input, y):
        """
        Backpropagation function.
        Returns the gradient of the cost function (MSE - Mean Squared Error) for a certain input

        :param input: np.ndarray of shape (n, 1), where n = self.sizes[0] (size of input layer)
        :param y: np.ndarray of shape (m, 1), where m = self.sizes[-1] (size of output layer)
        :returns: gradient in terms of both weights and biases, w.r.t. the provided input
        """
        # forward propagation - populating the neurons with their unactivated output
        z = [None for i in range(self.num_layers)]
        z[0] = np.array(input)  # call copy constructor
        for i in range(1, self.num_layers):
            a = sigmoid(z[i - 1])  # activate previous layer
            z[i] = np.dot(self.weights[i], a) + self.biases[i]

        # populating the errors in each layer (except the input one)
        errors = [None for i in range(self.num_layers)]
        errors[-1] = (sigmoid(z[-1]) - y) * derivative_sigmoid(z[-1])  # error in the output layer
        for i in range(self.num_layers - 2, 0, -1):
            errors[i] = np.dot(self.weights[i + 1].T, errors[i + 1]) * derivative_sigmoid(z[i])  # errors in the subsequent layers

        gradient_biases = [None for i in range(self.num_layers)]
        gradient_weights = [None for i in range(self.num_layers)]

        for i in range(1, self.num_layers):
            gradient_biases[i] = np.array(errors[i])  # call copy constructor

            # NOTE: construction of gradient_weights:
            # for each layer i, consider errors[i] and sigmoid(z[i - 1])
            # denote err = errors[i] and a = sigmoid(z[i - 1])
            # gradient_weights[i][j][k] = a[k] * err[j]
            # to do this without for loops, we can:
            # extend err to have the number of columns as the dimension of a (which is self.sizes[i - 1])
            # diagonalize a
            # multiply err with a to get the desired weight matrix

            err = np.array(errors[i])
            for j in range(self.sizes[i - 1] - 1):
                err = np.insert(err, 0, err[:, 0], axis=1)

            a = sigmoid(z[i - 1])
            a = np.diag(a[:, 0])  # where a[:, k] is the numpy array containing the elements of column k from the 2D - array a

            gradient_weights[i] = np.dot(err, a)

        return gradient_biases, gradient_weights

    def weights_biases_to_csv(self, path: str):
        for i in range(1, self.num_layers):
            biases = pd.DataFrame(self.biases[i])
            biases.to_csv(f'{path}/biases[{i}].csv', encoding='utf-8', index=False, header=False)
            weights = pd.DataFrame(self.weights[i])
            # weights.index = [f'weights[{i}][{j}]' for j in range(self.sizes[i])]
            weights.to_csv(f'{path}/weights[{i}].csv', encoding='utf-8', index=False, header=False)

    def SDG(self, mini_batch_size, epochs, learning_rate, training_data=None):
        """
        Stochastic Gradient Descent

        :param mini_batch_size: int
        :param epochs: int
        :learning_rate: float
        :training_data: None or numpy.ndarray of shape (n, m), where n = total number of training examples,
                                                                     m = self.sizes[0] + 1 (size of input layer + 1 for the label)
        :returns: None (stdout)
        """
        if training_data is None:
            training_data = mnist_train_df
            training_data = training_data.to_numpy()

        for epoch in range(epochs):
            np.random.shuffle(training_data)
            total_training_examples = training_data.shape[0]
            batches = total_training_examples // mini_batch_size

            for batch in range(batches):
                gradient_biases_average = [None] + [np.zeros((size, 1)) for size in self.sizes[1:]]
                gradient_weights_average = [None] + [np.zeros((self.sizes[i + 1], self.sizes[i])) for i in range(self.num_layers - 1)]

                for i in range(batch * mini_batch_size, (batch + 1) * mini_batch_size):
                    input = np.array(training_data[i][1:])  # position [i][0] is label
                    input = input[..., None]  # transforming 1D array into (n, 1) ndarray

                    y = self.get_expected_output(training_data[i][0])
                    gradient_biases_current, gradient_weights_current = self.backprop(input, y)

                    for i in range(1, self.num_layers):
                        gradient_biases_average[i] += gradient_biases_current[i]
                        gradient_weights_average[i] += gradient_weights_current[i]

                for i in range(1, self.num_layers):
                    gradient_biases_average[i] /= mini_batch_size
                    gradient_weights_average[i] /= mini_batch_size
                    self.biases[i] -= learning_rate * gradient_biases_average[i]
                    self.weights[i] -= learning_rate * gradient_weights_average[i]

            # NOTE: range of inputs if total_training_examples % mini_batch_size != 0: range(batches * mini_batch_size, total_training_examples)
            # number of training inputs: total_training_examples % mini_batch_size
            if total_training_examples % mini_batch_size != 0:
                gradient_biases_average = [None] + [np.zeros((size, 1)) for size in self.sizes[1:]]
                gradient_weights_average = [None] + [np.zeros((self.sizes[i + 1], self.sizes[i])) for i in range(self.num_layers - 1)]

                for i in range(batches * mini_batch_size, total_training_examples):
                    input = np.array(training_data[i][1:])  # position 0 is label
                    input = input[..., None]  # transforming 1D array into (n, 1) ndarray

                    y = self.get_expected_output(training_data[i][0])
                    gradient_biases_current, gradient_weights_current = self.backprop(input, y)

                    for i in range(1, self.num_layers):
                        gradient_biases_average[i] += gradient_biases_current[i]
                        gradient_weights_average[i] += gradient_weights_current[i]

                for i in range(1, self.num_layers):
                    gradient_biases_average[i] /= (total_training_examples % mini_batch_size)
                    gradient_weights_average[i] /= (total_training_examples % mini_batch_size)
                    self.biases[i] -= learning_rate * gradient_biases_average[i]
                    self.weights[i] -= learning_rate * gradient_weights_average[i]

            # test the network in each epoch
            print(f'Epoch {epoch}: ', end='')
            self.test_network()


digit_recognizer = Network([784, 64, 10])
digit_recognizer.SDG(20, 20, 1.5)
digit_recognizer.weights_biases_to_csv('../weights_biases/')
