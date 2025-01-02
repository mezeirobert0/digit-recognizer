from layer import Layer


class ActivationLayer(Layer):
    def __init__(self, activation, derivative_activation):
        self.activation = activation
        self.derivative_activation = derivative_activation

    def forward(self, input_array):
        self.input_array = input_array.copy()
        return self.activation(input_array)

    def backwards(self, output_gradient):
        return output_gradient * self.derivative_activation(self.input_array)
