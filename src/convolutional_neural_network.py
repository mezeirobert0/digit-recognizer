from concurrent.futures import ThreadPoolExecutor
from time import time
import numpy as np
from pathlib import Path
from activation_layer import ActivationLayer
from average_pooling_layer import AveragePoolingLayer
from convolutional_layer import ConvolutionalLayer
from dense_layer import DenseLayer
from layer import Layer
from reshape_layer import ReshapeLayer
from softmax_layer import SoftmaxLayer
from multiprocessing import Pool
from utils import (
    categorical_cross_entropy_loss,
    derivative_categorical_cross_entropy_loss,
    derivative_tanh,
    get_result,
    load_mnist,
    one_hot_encode,
    tanh,
    step_learning_rate_decay,
    preprocess_mnist_datapoint
)

np.random.seed(0)

mnist_train, mnist_test = load_mnist()

# lenet_5 = [
#     ConvolutionalLayer((1, 32, 32), (6, 28, 28)),   # 0
#     ActivationLayer(tanh, derivative_tanh),
#     AveragePoolingLayer(2),
#     ConvolutionalLayer((6, 14, 14), (16, 10, 10)),  # 3
#     ActivationLayer(tanh, derivative_tanh),
#     AveragePoolingLayer(2),
#     ConvolutionalLayer((16, 5, 5), (120, 1, 1)),    # 6
#     ActivationLayer(tanh, derivative_tanh),
#     ReshapeLayer((120, 1, 1), (120, 1)),
#     DenseLayer(120, 84),                            # 9
#     ActivationLayer(tanh, derivative_tanh),
#     DenseLayer(84, 10),                             # 11
#     SoftmaxLayer(),
# ]

# train_network(lenet_5, epochs=6, mini_batch_size=5, learning_rate=0.01)
# trainable_parameters_to_csv(lenet_5)

# lenet_5_pretrained = [
#     ConvolutionalLayer((1, 32, 32), (6, 28, 28), '../data/kernels_biases/layer_0'),   # 0
#     ActivationLayer(tanh, derivative_tanh),
#     AveragePoolingLayer(2),
#     ConvolutionalLayer((6, 14, 14), (16, 10, 10), '../data/kernels_biases/layer_3'),  # 3
#     ActivationLayer(tanh, derivative_tanh),
#     AveragePoolingLayer(2),
#     ConvolutionalLayer((16, 5, 5), (120, 1, 1), '../data/kernels_biases/layer_6'),    # 6
#     ActivationLayer(tanh, derivative_tanh),
#     ReshapeLayer((120, 1, 1), (120, 1)),
#     DenseLayer(120, 84, '../data/weights_biases/layer_9'),                            # 9
#     ActivationLayer(tanh, derivative_tanh),
#     DenseLayer(84, 10, '../data/weights_biases/layer_11'),                            # 11
#     SoftmaxLayer(),
# ]
# test_network(lenet_5_pretrained)
# test_network_async(lenet_5_pretrained)
# train_network(lenet_5_pretrained, epochs=5, mini_batch_size=5, learning_rate=0.008)
# trainable_parameters_to_csv(lenet_5_pretrained)

class ConvolutionalNeuralNetwork:
    def __init__(self, layers: list[Layer]):
        self.layers = layers

    def fit(self, epochs: int, mini_batch_size: int, learning_rate: float, step: int, decay_rate: float, training_data: np.ndarray = None):
        if training_data is None:
            training_data = mnist_train.copy()
        total_training = training_data.shape[0]
        batches = total_training // mini_batch_size

        for epoch in range(epochs):
            np.random.shuffle(training_data)
            error = 0

            start_time = time()

            for batch in range(batches):
                for i in range(batch * mini_batch_size, (batch + 1) * mini_batch_size):
                    if (i + 1) % 1000 == 0:
                        print(i + 1)

                    expected_output_array = one_hot_encode(training_data[i, 0], 10)
                    output_array = preprocess_mnist_datapoint(training_data[i, 1:])

                    # forward propagation
                    for layer in self.layers:
                        output_array = layer.forward(output_array)

                    error += categorical_cross_entropy_loss(expected_output_array, output_array)

                    # backwards propagation
                    gradient = derivative_categorical_cross_entropy_loss(expected_output_array, output_array)
                    for layer in reversed(self.layers):
                        gradient = layer.backwards(gradient)

                for layer in self.layers:
                    if isinstance(layer, ConvolutionalLayer):
                        layer.update_kernels_biases(learning_rate)
                    elif isinstance(layer, DenseLayer):
                        layer.update_weights_biases(learning_rate)

            error /= total_training
            print(f"Error in training dataset: {error}")
            print(f'Epoch {epoch}: ', end='')
            self.test_network_async()
            end_time = time()
            print(f'Time took for epoch {epoch}: {end_time - start_time} seconds')
            print()

            # update learning rate
            learning_rate = step_learning_rate_decay(learning_rate, epoch, step, decay_rate)

    def test_network_async(self, testing_data: np.ndarray = None, num_threads: int = 2):
        if testing_data is None:
            testing_data = mnist_test.copy()
        total_testing = testing_data.shape[0]

        # start_time = time()
        total_predicted = 0
        batch_size = total_testing // num_threads
        with Pool(num_threads) as pool:
            results = pool.starmap(self.test_network_batch, [(testing_data, i * batch_size, (i + 1) * batch_size) for i in range(num_threads)])
            total_predicted = sum(results)

        # end_time = time()
        print(f'{total_predicted} / {total_testing}')
        # print(f'Time taken: {end_time - start_time} seconds')

    def test_network_batch(self, testing_data: np.ndarray, start: int, end: int):
        total_predicted = 0
        for i in range(start, end):
            expected_result = testing_data[i, 0]
            output_array = preprocess_mnist_datapoint(testing_data[i, 1:])

            for layer in self.layers:
                output_array = layer.forward(output_array)

            prediction = get_result(output_array)
            if prediction == expected_result:
                total_predicted += 1

        return total_predicted
    
    def trainable_parameters_to_csv(self):
        for index, layer in enumerate(self.layers):
            if isinstance(layer, ConvolutionalLayer):
                layer.kernels_biases_to_csv(f'../data/kernels_biases/layer_{index}/')
            elif isinstance(layer, DenseLayer):
                layer.weights_biases_to_csv(f'../data/weights_biases/layer_{index}/')

    def get_prediction_confidence(self, image: np.ndarray) -> tuple[int, float]: 
        output_array = image
        for layer in self.layers:
            output_array = layer.forward(output_array)

        prediction = get_result(output_array)
        confidence = output_array[prediction, 0]

        return prediction, confidence
    
def get_lenet_5_pretrained():
    lenet_5_pretrained = ConvolutionalNeuralNetwork([
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
    ])

    # lenet_5_pretrained.test_network_async()
    return lenet_5_pretrained 
    
if __name__ == '__main__':
    # lenet_5 = ConvolutionalNeuralNetwork([
    #     ConvolutionalLayer((1, 32, 32), (6, 28, 28)),   # 0
    #     ActivationLayer(tanh, derivative_tanh),
    #     AveragePoolingLayer(2),
    #     ConvolutionalLayer((6, 14, 14), (16, 10, 10)),  # 3
    #     ActivationLayer(tanh, derivative_tanh),
    #     AveragePoolingLayer(2),
    #     ConvolutionalLayer((16, 5, 5), (120, 1, 1)),    # 6
    #     ActivationLayer(tanh, derivative_tanh),
    #     ReshapeLayer((120, 1, 1), (120, 1)),
    #     DenseLayer(120, 84),                            # 9
    #     ActivationLayer(tanh, derivative_tanh),
    #     DenseLayer(84, 10),                             # 11
    #     SoftmaxLayer(),
    # ])

    # lenet_5.fit(epochs=7, mini_batch_size=5, learning_rate=0.01, step=1, decay_rate=0.9)
    # lenet_5.trainable_parameters_to_csv()

    lenet_5_pretrained = ConvolutionalNeuralNetwork([
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
    ])

    lenet_5_pretrained.test_network_async()
