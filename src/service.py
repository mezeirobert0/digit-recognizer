import numpy as np
from mnist import Mnist
from convolutional_neural_network import get_lenet_5_pretrained
from utils import preprocess_mnist_datapoint


class Service:
    def __init__(self):
        self.lenet_5_pretrained = get_lenet_5_pretrained()
        
    def get_prediction_confidence(self, input_array: np.ndarray):
        """
        Get prediction and confidence
        :param input_array: np.ndarray of shape (28, 28)
        :returns: tuple with the prediction (0-9) and accuracy (%)
        """
        # normalize and standardize input_array
        mean = self.lenet_5_pretrained.mnist_instance.get_X_train_mean()
        std = self.lenet_5_pretrained.mnist_instance.get_X_train_std()
        input_array = ((input_array / 255) - mean) / std

        return self.lenet_5_pretrained.get_prediction_confidence(preprocess_mnist_datapoint(input_array))
