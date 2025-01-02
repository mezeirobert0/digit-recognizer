import numpy as np
from layer import Layer


class SoftmaxLayer(Layer):
    def forward(self, input_array):
        max_val = np.max(input_array)  # max value needed for stable softmax
        tmp = np.exp(input_array - max_val)
        self.output_array = tmp / np.sum(tmp)
        return self.output_array.copy()
    
    def backwards(self, output_gradient):
        n = np.size(self.output_array)
        tmp = np.tile(self.output_array, n)
        return np.dot(tmp * (np.identity(n) - tmp.T), output_gradient)