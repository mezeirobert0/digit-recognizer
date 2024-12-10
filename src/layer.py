import numpy as np

class Layer:
    def __init__(self):
        self.input_vector = None
        self.output_vector = None

    def forward(self, input_vector: np.ndarray):
        pass

    def backwards(self, output_gradient: np.ndarray, learning_rate: float):
        pass
