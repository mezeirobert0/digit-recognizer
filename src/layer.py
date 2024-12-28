import numpy as np

class Layer:
    def __init__(self):
        self.input_array = None
        self.output_array = None

    def forward(self, input_array: np.ndarray):
        pass

    def backwards(self, output_gradient: np.ndarray, learning_rate: float):
        pass
