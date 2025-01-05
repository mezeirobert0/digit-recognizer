import pandas as pd
import numpy as np
from pathlib import Path
from layer import Layer
    
class DenseLayer(Layer):
    def __init__(self, input_size: int, output_size: int, path: str = None):
        if path is None:
            Fi = input_size  # Fan-in initialization
            self.weights = np.random.uniform(-2.4/Fi, 2.4/Fi, (output_size, input_size))
            self.biases = np.ones((output_size, 1)) * 0.01
        
        else:
            self.biases = pd.read_csv(f'{path}/biases.csv', header=None).to_numpy()
            self.weights = pd.read_csv(f'{path}/weights.csv', header=None).to_numpy()

        self.sum_biases_gradient = None
        self.sum_weights_gradient = None
        self.total_gradients = 0

    def forward(self, input_array):
        self.input_array = input_array.copy()
        return np.dot(self.weights, self.input_array) + self.biases
    
    def backwards(self, output_gradient):
        if self.total_gradients == 0:
            self.sum_biases_gradient = np.zeros(self.biases.shape)
            self.sum_weights_gradient = np.zeros(self.weights.shape)
        
        weights_gradient = np.dot(output_gradient, self.input_array.T)
        biases_gradient = output_gradient.copy()
        input_gradient = np.dot(self.weights.T, output_gradient)

        self.total_gradients += 1
        self.sum_biases_gradient += biases_gradient
        self.sum_weights_gradient += weights_gradient

        # return gradient of cost function w.r.t. input
        return input_gradient
    
    def update_weights_biases(self, learning_rate: float):
        self.biases -= learning_rate / self.total_gradients * self.sum_biases_gradient
        self.weights -= learning_rate / self.total_gradients * self.sum_weights_gradient

        # reset the sum of gradients of weights and biases
        self.sum_biases_gradient = None
        self.sum_weights_gradient = None
        self.total_gradients = 0
    
    def weights_biases_to_csv(self, path: str):
        weights = pd.DataFrame(self.weights)
        filepath = Path(f'{path}/weights.csv')
        filepath.parent.mkdir(parents=True, exist_ok=True)
        weights.to_csv(filepath, encoding="utf-8", index=False, header=False)

        biases = pd.DataFrame(self.biases)
        filepath = Path(f'{path}/biases.csv')
        filepath.parent.mkdir(parents=True, exist_ok=True)
        biases.to_csv(filepath, encoding="utf-8", index=False, header=False)
