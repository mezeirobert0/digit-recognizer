from layer import Layer
import numpy as np

class ActivationLayer(Layer):
    def __init__(self, activation, derivative_activation):
        self.activation = activation
        self.derivative_activation = derivative_activation

    def forward(self, input_vector) -> np.ndarray:
        self.input_vector = input_vector.copy()
        return self.activation(input_vector)
    
    def backwards(self, output_gradient, learning_rate) -> np.ndarray:
        return output_gradient * self.derivative_activation(self.input_vector)