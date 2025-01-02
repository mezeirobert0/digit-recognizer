from convolutional_neural_network import get_lenet_5_pretrained
from utils import preprocess_mnist_datapoint


class Service:
    def __init__(self):
        self.lenet_5_pretrained = get_lenet_5_pretrained()
        
    def get_prediction_confidence(self, input_array):
        """
        Get prediction and accuracy
        :param input_array: np.ndarray of shape (28, 28)
        :returns: int, float
        """
        return self.lenet_5_pretrained.get_prediction_confidence(preprocess_mnist_datapoint(input_array))
