import numpy as np
from layer import Layer
    
class DenseLayer(Layer):
    def __init__(self, input_size: int, output_size: int):
        self.biases = np.random.randn(output_size, 1)
        self.weights = np.random.randn(output_size, input_size)

    def forward(self, input_vector):
        self.input_vector = input_vector.copy()
        return np.dot(self.weights, self.input_vector) + self.biases
    
    def backwards(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.input_vector.T)
        biases_gradient = output_gradient.copy()
        input_gradient = np.dot(self.weights.T, output_gradient)

        # update weights and biases
        self.weights -= learning_rate * weights_gradient
        self.biases -= learning_rate * biases_gradient

        # return gradient of cost function w.r.t. input
        return input_gradient.copy()


