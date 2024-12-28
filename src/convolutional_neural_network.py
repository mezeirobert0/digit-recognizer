import numpy as np
from pathlib import Path
from activation_layer import ActivationLayer
from average_pooling_layer import AveragePoolingLayer
from convolutional_layer import ConvolutionalLayer
from dense_layer import DenseLayer
from layer import Layer
from reshape_layer import ReshapeLayer
from softmax_layer import SoftmaxLayer
from utils import (
    categorical_cross_entropy_loss,
    derivative_categorical_cross_entropy_loss,
    derivative_tanh,
    get_result,
    load_mnist,
    one_hot_encode,
    tanh,
)

np.random.seed(0)

mnist_train, mnist_test = load_mnist()

training_data = mnist_train.copy()
testing_data = mnist_test.copy()

total_training = training_data.shape[0]
total_testing = testing_data.shape[0]

def train_network(network: list[Layer], epochs: int, learning_rate: float):
    for epoch in range(epochs):
        np.random.shuffle(training_data)
        error = 0

        for i in range(total_training):
            if (i + 1) % 1000 == 0:
                print(i + 1)

            expected_output_array = one_hot_encode(training_data[i][0], 10)
            output_array = training_data[i][1:] / 255
            output_array = np.pad(output_array.reshape(28, 28), 2)[None, ...]  # reshape, pad with 2 on each side, add extra dimension

            # forward propagation
            for layer in network:
                output_array = layer.forward(output_array)

            error += categorical_cross_entropy_loss(expected_output_array, output_array)

            # backwards propagation
            gradient = derivative_categorical_cross_entropy_loss(expected_output_array, output_array)
            for layer in reversed(network):
                gradient = layer.backwards(gradient, learning_rate)

        error /= total_training
        print(f"Error in training dataset: {error}")
        print(f'Epoch {epoch}: ', end='')
        test_network(network)
        print()

def test_network(network: list[Layer]):
    total_predicted = 0
    # test the network after each epoch
    for i in range(total_testing):
        expected_result = testing_data[i][0]
        output_array = testing_data[i][1:] / 255
        output_array = np.pad(output_array.reshape(28, 28), 2)[None, ...]  # reshape, pad with 2 on each side, add extra dimension

        # forward propagation
        for layer in network:
            output_array = layer.forward(output_array)

        prediction = get_result(output_array)

        if prediction == expected_result:
            total_predicted += 1

    print(f'{total_predicted} / {total_testing}')


def trainable_parameters_to_csv(network: list[Layer]):
    for index, layer in enumerate(network):
        if isinstance(layer, ConvolutionalLayer):
            layer.kernels_biases_to_csv(f'../data/kernels_biases/layer_{index}/')
        elif isinstance(layer, DenseLayer):
            layer.weights_biases_to_csv(f'../data/weights_biases/layer_{index}/')

lenet_5 = [
    ConvolutionalLayer((1, 32, 32), (6, 28, 28)),   # 0
    ActivationLayer(tanh, derivative_tanh),
    AveragePoolingLayer(2),
    ConvolutionalLayer((6, 14, 14), (16, 10, 10)),  # 3
    ActivationLayer(tanh, derivative_tanh),
    AveragePoolingLayer(2),
    ConvolutionalLayer((16, 5, 5), (120, 1, 1)),    # 6
    ActivationLayer(tanh, derivative_tanh),
    ReshapeLayer((120, 1, 1), (120, 1)),
    DenseLayer(120, 84),                            # 9
    ActivationLayer(tanh, derivative_tanh),
    DenseLayer(84, 10),                             # 11
    SoftmaxLayer(),
]

train_network(lenet_5, 1, 0.01)
trainable_parameters_to_csv(lenet_5)

lenet_5_pretrained = [
    ConvolutionalLayer((1, 32, 32), (6, 28, 28), '../data/kernels_biases/layer_0'),   # 0
    ActivationLayer(tanh, derivative_tanh),
    AveragePoolingLayer(2),
    ConvolutionalLayer((6, 14, 14), (16, 10, 10), '../data/kernels_biases/layer_3'),  # 3
    ActivationLayer(tanh, derivative_tanh),
    AveragePoolingLayer(2),
    ConvolutionalLayer((16, 5, 5), (120, 1, 1), '../data/kernels_biases/layer_6'),    # 6
    ActivationLayer(tanh, derivative_tanh),
    ReshapeLayer((120, 1, 1), (120, 1)),
    DenseLayer(120, 84, '../data/weights_biases/layer_9'),                            # 9
    ActivationLayer(tanh, derivative_tanh),
    DenseLayer(84, 10, '../data/weights_biases/layer_11'),                            # 11
    SoftmaxLayer(),
]
test_network(lenet_5_pretrained)
