import pandas as pd
import numpy as np
from pathlib import Path
from layer import Layer
    
class DenseLayer(Layer):
    def __init__(self, input_size: int, output_size: int, path: str = None):
        if path is None:
            self.biases = np.random.randn(output_size, 1)
            self.weights = np.random.randn(output_size, input_size)
        
        else:
            self.biases = pd.read_csv(f'{path}/biases.csv', header=None).to_numpy()
            self.weights = pd.read_csv(f'{path}/weights.csv', header=None).to_numpy()

    def forward(self, input_array):
        self.input_array = input_array.copy()
        return np.dot(self.weights, self.input_array) + self.biases
    
    def backwards(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.input_array.T)
        biases_gradient = output_gradient.copy()
        input_gradient = np.dot(self.weights.T, output_gradient)

        # update weights and biases
        self.weights -= learning_rate * weights_gradient
        self.biases -= learning_rate * biases_gradient

        # return gradient of cost function w.r.t. input
        return input_gradient.copy()
    
    def weights_biases_to_csv(self, path: str):
        weights = pd.DataFrame(self.weights)
        filepath = Path(f'{path}/weights.csv')
        filepath.parent.mkdir(parents=True, exist_ok=True)
        weights.to_csv(filepath, encoding="utf-8", index=False, header=False)

        biases = pd.DataFrame(self.biases)
        filepath = Path(f'{path}/biases.csv')
        filepath.parent.mkdir(parents=True, exist_ok=True)
        biases.to_csv(filepath, encoding="utf-8", index=False, header=False)
