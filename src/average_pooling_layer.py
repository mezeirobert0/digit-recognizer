import numpy as np
from layer import Layer

class AveragePoolingLayer(Layer):
    def __init__(self, pool_size: int):
        self.pool_size = pool_size

    def forward(self, input_array):
        self.input_array = input_array.copy()
        num_channels, input_height, input_width = input_array.shape
        output_height = input_height // self.pool_size
        output_width = input_width // self.pool_size

        self.output_array = np.zeros((num_channels, output_height, output_width))

        for c in range(num_channels):
            for i in range(output_height):
                start_i = i * self.pool_size
                end_i = start_i + self.pool_size

                for j in range(output_width):
                    start_j = j * self.pool_size
                    end_j = start_j + self.pool_size

                    self.output_array[c, i, j] = np.average(input_array[c, start_i:end_i, start_j:end_j])
        
        return self.output_array.copy()

    def backwards(self, output_gradient: np.ndarray, learning_rate):
        input_gradient = np.zeros(self.input_array.shape)
        num_channels, input_height, input_width = self.input_array.shape
        output_height = input_height // self.pool_size
        output_width = input_width // self.pool_size

        for c in range(num_channels):
            for i in range(output_height):
                start_i = i * self.pool_size
                end_i = start_i + self.pool_size

                for j in range(output_width):
                    start_j = j * self.pool_size
                    end_j = start_j + self.pool_size

                    input_gradient[c, start_i:end_i, start_j:end_j].fill(output_gradient[c, i, j] / self.pool_size ** 2)

        return input_gradient.copy()


'''
layer1 = AveragePoolingLayer(2)
input_arr = np.array([[
    [2, 12, 5, 10, 2, 7],
    [9, 16, 8, 17, 5, 7],
    [14, 3, 8, 2, 16, 1],
    [2, 0, 17, 10, 8, 11],
    [19, 5, 7, 0, 14, 16],
    [1, 1, 20, 12, 0, 8]
]])
print(layer1.forward(input_arr))
print()
print(layer1.backwards(layer1.forward(input_arr)))
'''