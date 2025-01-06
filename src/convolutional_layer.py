import pandas as pd
import numpy as np
from pathlib import Path
from layer import Layer
from scipy.signal import correlate2d, convolve2d


class ConvolutionalLayer(Layer):
    def __init__(self, input_shape: tuple, output_shape: tuple, path: str = None):
        """Initialization of biases and kernels

        :param input_shape: tuple of 3 integers (number of channels, input image height, input image width)
        :param output_shape: tuple of 3 integers (number of output images, output image height, output image width)
        """
        self.input_shape = input_shape
        self.output_shape = output_shape

        if path is None:
            kernels_shape = (
                output_shape[0],
                input_shape[0],
                input_shape[1] - output_shape[1] + 1,
                input_shape[2] - output_shape[2] + 1
            )

            Fi = input_shape[0] * kernels_shape[2] * kernels_shape[3]  # Fan-in initialization
            self.kernels = np.random.uniform(-2.4/Fi, 2.4/Fi, kernels_shape)
            self.biases = np.ones(output_shape) * 0.01
            

        else:
            self.biases = np.zeros(output_shape)
            self.kernels = np.zeros((
                output_shape[0],
                input_shape[0],
                input_shape[1] - output_shape[1] + 1,
                input_shape[2] - output_shape[2] + 1
            ))

            for i in range(self.kernels.shape[0]):
                for j in range(self.kernels.shape[1]):
                    self.kernels[i, j] = pd.read_csv(f"{path}/kernel_{i}_{j}.csv", encoding="utf-8", header=None).to_numpy()

            for i in range(self.biases.shape[0]):
                self.biases[i] = pd.read_csv(f"{path}/bias_{i}.csv", encoding="utf-8", header=None).to_numpy()

        self.sum_biases_gradient = None
        self.sum_kernels_gradient = None
        self.total_gradients = 0
    
    def forward(self, input_array: np.ndarray):
        self.input_array = input_array.copy()
        self.output_array = np.zeros(self.output_shape)

        for i in range(self.output_shape[0]):
            sum_correlations = np.zeros(self.output_shape[1:])
            for j in range(self.input_shape[0]):
                sum_correlations += correlate2d(self.input_array[j], self.kernels[i, j], 'valid')
            
            self.output_array[i] = sum_correlations + self.biases[i]

        return self.output_array.copy()
    
    def backwards(self, output_gradient: np.ndarray):
        if self.total_gradients == 0:
            self.sum_biases_gradient = np.zeros(self.biases.shape)
            self.sum_kernels_gradient = np.zeros(self.kernels.shape)

        biases_gradient = output_gradient.copy()
        kernels_gradient = np.zeros(self.kernels.shape)
        input_gradient = np.zeros(self.input_shape)
        for i in range(self.output_shape[0]):
            for j in range(self.input_shape[0]):
                kernels_gradient[i, j] = correlate2d(self.input_array[j], output_gradient[i], 'valid')
                input_gradient[j] += convolve2d(output_gradient[i], self.kernels[i, j], 'full')

        self.total_gradients += 1
        self.sum_biases_gradient += biases_gradient
        self.sum_kernels_gradient += kernels_gradient

        return input_gradient
    
    def update_kernels_biases(self, learning_rate: float):
        self.biases -= learning_rate / self.total_gradients * self.sum_biases_gradient
        self.kernels -= learning_rate / self.total_gradients * self.sum_kernels_gradient

        # reset the sum of gradients of kernels and biases
        self.sum_biases_gradient = None
        self.sum_kernels_gradient = None
        self.total_gradients = 0
    
    def kernels_biases_to_csv(self, path: str):
        for i in range(self.kernels.shape[0]):
            for j in range(self.kernels.shape[1]):
                kernel = pd.DataFrame(self.kernels[i, j])
                filepath = Path(f'{path}/kernel_{i}_{j}.csv')
                filepath.parent.mkdir(parents=True, exist_ok=True)
                kernel.to_csv(filepath, encoding="utf-8", index=False, header=False)

        for i in range(self.biases.shape[0]):
            bias = pd.DataFrame(self.biases[i])
            filepath = Path(f'{path}/bias_{i}.csv')
            filepath.parent.mkdir(parents=True, exist_ok=True)
            bias.to_csv(filepath, encoding="utf-8", index=False, header=False)
