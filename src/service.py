import network


class Service:
    def __init__(self):
        self.network = network.Network([784, 64, 10], "../data/weights_biases/")

    def get_prediction(self, input_vector):
        """
        Get prediction
        :param input_vector: np.ndarray of shape (n, 1), normalized
        :returns: int
        """
        return int(self.get_result(self.feedforward(input_vector)))
