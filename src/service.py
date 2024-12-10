import fully_connected_network


class Service:
    def __init__(self):
        self.network = fully_connected_network.FullyConnectedNetwork([784, 64, 10], "../data/weights_biases/")

    def get_prediction_confidence(self, input_vector):
        """
        Get prediction and accuracy
        :param input_vector: np.ndarray of shape (n, 1), normalized
        :returns: int, float
        """
        output = self.network.feedforward(input_vector)
        predicted_result = int(self.network.get_result(output))
        confidence = output[predicted_result][0]
        return predicted_result, confidence
