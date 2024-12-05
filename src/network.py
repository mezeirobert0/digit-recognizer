import numpy as np
import pandas as pd
np.random.seed(0)


def ReLU(z):
    return np.maximum(z, 0)


def derivative_ReLU(z):
    return z > 0


def softmax(z):
    exp = np.exp(z)
    return exp / sum(exp)


mnist_train_df = pd.read_csv("../data/datasets/mnist_train.csv")
mnist_test_df = pd.read_csv("../data/datasets/mnist_test.csv")


class Network:
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

    def feedforward(self, input):
        """
        Returns the output of the network, given a certain input
        :param input: np.ndarray of shape (n, 1), where n = self.sizes[0] (size of input layer)
        :returns: np.ndarray of shape (m, 1), where m = self.sizes[-1] (size of output layer)
        """
        x = np.array(input)  # call copy constructor
        for i in range(1, self.num_layers - 1):
            x = ReLU(np.dot(self.weights[i], x) + self.biases[i])
        x = softmax(np.dot(self.weights[-1], x) + self.biases[-1])
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

    def get_expected_output(self, expected_result: int):
        """
        Returns the vector corresponding to the expected output of the network
        :param expected_result: int, between 0 and m - 1
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
        :returns: None
        """
        if testing_data is None:
            testing_data = mnist_test_df
            testing_data = testing_data.to_numpy()
        total_correct = 0
        total = testing_data.shape[0]
        incorrect = []
        for i in range(total):
            input_vector = testing_data[i][1:] / 255  # normalize input, label is on column 0
            input_vector = input_vector[..., None]  # transforming 1D array into (n, 1) ndarray
            if self.get_result(self.feedforward(input_vector)) == testing_data[i][0]:
                total_correct += 1
            else:
                incorrect.append(testing_data)
        print(f"{total_correct}/{total}")

    def backprop(self, input_vector, y):
        """
        Backpropagation function.
        Returns the gradient of the cost function (MSE - Mean Squared Error) for a certain input
        :param input: np.ndarray of shape (n, 1), where n = self.sizes[0] (size of input layer)
        :param y: np.ndarray of shape (m, 1), where m = self.sizes[-1] (size of output layer)
        :returns: gradient in terms of both weights and biases, w.r.t. the provided input
        """

        # forward propagation
        z = [None]
        a = [np.array(input_vector)]
        for i in range(1, self.num_layers - 1):
            z.append(np.dot(self.weights[i], a[-1]) + self.biases[i])
            a.append(ReLU(z[-1]))
        z.append(np.dot(self.weights[-1], a[-1]) + self.biases[-1])
        a.append(softmax(z[-1]))  # for the last layer apply softmax activation function instead of

        gradient_biases = [None] * self.num_layers
        gradient_weights = [None] * self.num_layers

        # backwards propagation
        error = (a[-1] - y)  # error in the output layer
        gradient_biases[-1] = np.array(error)
        gradient_weights[-1] = np.dot(error, a[-2].T)
        for i in range(self.num_layers - 2, 0, -1):
            error = np.dot(self.weights[i + 1].T, error) * derivative_ReLU(z[i])  # error in the subsequent layer
            gradient_biases[i] = np.array(error)
            gradient_weights[i] = np.dot(error, a[i - 1].T)

        return gradient_biases, gradient_weights

    def weights_biases_to_csv(self, path: str):
        for i in range(1, self.num_layers):
            biases = pd.DataFrame(self.biases[i])
            biases.to_csv(f"{path}/biases[{i}].csv", encoding="utf-8", index=False, header=False)
            weights = pd.DataFrame(self.weights[i])
            weights.to_csv(f"{path}/weights[{i}].csv", encoding="utf-8", index=False, header=False)

    # TODO: refactor code in this function
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
            training_data = mnist_train_df
            training_data = training_data.to_numpy()

        total_training_examples = training_data.shape[0]
        batches = total_training_examples // mini_batch_size

        for epoch in range(epochs):
            np.random.shuffle(training_data)

            for batch in range(batches):
                gradient_biases_sum = [None] + [np.zeros((size, 1)) for size in self.sizes[1:]]
                gradient_weights_sum = [None] + [np.zeros((self.sizes[i + 1], self.sizes[i])) for i in range(self.num_layers - 1)]

                for i in range(batch * mini_batch_size, (batch + 1) * mini_batch_size):
                    # print(f"Input {i}")
                    input_vector = np.array(training_data[i][1:]) / 255  # position [i][0] is label
                    input_vector = input_vector[..., None]  # transforming 1D array into (n, 1) ndarray

                    y = self.get_expected_output(training_data[i][0])
                    gradient_biases_current, gradient_weights_current = self.backprop(input_vector, y)

                    for i in range(1, self.num_layers):
                        gradient_biases_sum[i] += gradient_biases_current[i]
                        gradient_weights_sum[i] += gradient_weights_current[i]

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
                    input_vector = np.array(training_data[i][1:])  # position 0 is label
                    input_vector = input_vector[..., None]  # transforming 1D array into (n, 1) ndarray

                    y = self.get_expected_output(training_data[i][0])
                    gradient_biases_current, gradient_weights_current = self.backprop(input_vector, y)

                    for i in range(1, self.num_layers):
                        gradient_biases_sum[i] += gradient_biases_current[i]
                        gradient_weights_sum[i] += gradient_weights_current[i]

                for i in range(1, self.num_layers):
                    self.biases[i] -= (learning_rate / (total_training_examples % mini_batch_size)) * gradient_biases_sum[i]
                    self.weights[i] -= (learning_rate / (total_training_examples % mini_batch_size)) * gradient_weights_sum[i]

            # test the network in each epoch
            print(f"Epoch {epoch}: ", end="")
            self.test_network()


if __name__ == "__main__":
    # train the network
    digit_recognizer = Network([784, 64, 10])
    digit_recognizer.test_network()
    digit_recognizer.SGD(30, 20, 0.1)
    digit_recognizer.weights_biases_to_csv("../data/weights_biases/")
