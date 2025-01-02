import numpy as np
from layer import Layer

class ReshapeLayer(Layer):
    def __init__(self, input_shape: tuple, output_shape: tuple):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input_array):
        return np.reshape(input_array, self.output_shape)
    
    def backwards(self, output_gradient):
        return np.reshape(output_gradient, self.input_shape)