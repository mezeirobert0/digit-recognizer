import numpy as np


def sigmoid(z):
    return 1/(1 + np.exp(-z))


class Network():
    def __init__(self, sizes: list[int]):
        self.num_layers = len(sizes)
        self.sizes = sizes[:]

        # the biases are stored in a list of numpy arrays (column vectors):
        # the biases of the 2nd layer are stored in self.biases[0], the biases of the 3rd layer are stored in self.biases[1], etc.
        # all layers but the input layer get biases
        self.biases = [np.random.randn(size, 1) for size in sizes[1:]]

        # initializing weights: list of numpy arrays (matrices)
        # the weights between the i-th layer and the (i+1)-th layer are stored in a 2D numpy array at self.weights[i-1]
        self.weights = [np.random.randn(sizes[i+1], sizes[i])
                        for i in range(self.num_layers - 1)]

    def feedforward(self, x):
        """Return the output of the network when x is input"""
        for i in range(self.num_layers - 1):
            x = sigmoid(np.dot(self.weights[i], x) + self.biases[i])
        return x

    # TODO: write Stochastic Gradient Descent function


net = Network([4, 3, 2])
print((net.weights[0]))
print(sigmoid(4))
net.feedforward([1, 2, 3])
