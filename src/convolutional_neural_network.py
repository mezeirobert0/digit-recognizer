from time import time
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
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
    one_hot_encode,
    shuffle_data,
    tanh,
    step_learning_rate_decay,
)
from mnist import Mnist

np.random.seed(0)

# X_train_standardized, y_train, X_test_standardized, y_test 

class ConvolutionalNeuralNetwork:
    def __init__(self, layers: list[Layer], num_classes: int):
        self.layers = layers
        self.num_classes = num_classes
        self.train_errors = np.zeros((1,))
        self.train_accuracies = np.zeros((1,))
        self.mnist_instance = Mnist()

    def fit(self, epochs: int, mini_batch_size: int, learning_rate: float, step: int, decay_rate: float, X_train: np.ndarray = None, y_train: np.ndarray = None):
        """Fit the network's learnable parameters (kernels, weights and biases) to the training data.
        Implemented optimizer: mini-batch gradient descent.
        """
        if X_train is None or y_train is None:
            X_train, y_train = self.mnist_instance.get_X_y_train()

        total_training = X_train.shape[0]
        batches = total_training // mini_batch_size

        # allocate arrays for errors and accuracies
        # from both the training and testing datasets
        self.train_errors = np.zeros((epochs,))
        self.train_accuracies = np.zeros((epochs,))
        self.test_errors = np.zeros((epochs,))
        self.test_accuracies = np.zeros((epochs,))

        for epoch in range(epochs):
            # shuffle the training data in each epoch
            # helps to avoid the loss function being stuck in a shallow, local minimum
            X_train, y_train = shuffle_data(X_train, y_train)

            print(f"Epoch {epoch + 1}")
            for batch in tqdm(range(batches), ncols=80):
                for i in range(batch * mini_batch_size, (batch + 1) * mini_batch_size):
                    expected_output_array = one_hot_encode(y_train[i], self.num_classes)
                    output_array = Mnist.preprocess_mnist_datapoint(X_train[i])

                    # forward propagation
                    for layer in self.layers:
                        output_array = layer.forward(output_array)

                    # backwards propagation
                    gradient = derivative_categorical_cross_entropy_loss(expected_output_array, output_array)
                    for layer in reversed(self.layers):
                        gradient = layer.backwards(gradient)

                # after backpropagating the error of the network with respect to each input in the mini-batch
                # the gradients of the loss function with respect to each set of kernels/weights/biases
                # have accumulated in each layer and now it's time to update these kernels/weights/biases
                for layer in self.layers:
                    if isinstance(layer, ConvolutionalLayer):
                        layer.update_kernels_biases(learning_rate)
                    elif isinstance(layer, DenseLayer):
                        layer.update_weights_biases(learning_rate)

            # in order to gain insights (overfitting, underfitting, etc.) on the network's performance
            # we need the accuracy and overall error (loss) from testing the network
            # on both the training and testing datasets
            total_predicted_training, _, train_error = self.test_network_async(X_train, y_train)
            total_predicted_testing, total_testing, test_error = self.test_network_async()

            print(f"Error in training dataset: {train_error}")
            print(f"Error in testing dataset: {test_error}")

            print(f'Accuracy in training dataset: {total_predicted_training} / {total_training}')
            print(f"Accuracy in testing dataset: {total_predicted_testing} / {total_testing}", end='\n\n')

            # save the overall error and accuracy from each epoch
            # in order to further assess the performance of the model
            self.train_errors[epoch] = train_error
            self.train_accuracies[epoch] = total_predicted_training / total_training
            self.test_errors[epoch] = test_error
            self.test_accuracies[epoch] = total_predicted_testing / total_testing

            # update learning rate
            learning_rate = step_learning_rate_decay(learning_rate, epoch, step, decay_rate)

    def test_network_async(self, X_test: np.ndarray = None, y_test: np.ndarray = None, num_processes: int = 2):
        """Test multiple batches of the dataset in parallel"""
        if X_test is None or y_test is None:
            X_test, y_test = self.mnist_instance.get_X_y_test()
        total_testing = X_test.shape[0]

        total_predicted = 0
        total_error = 0

        batch_size = total_testing // num_processes
        with Pool(num_processes) as pool:
            results = pool.starmap(
                self.test_network_batch,
                [(X_test[i * batch_size:(i + 1) * batch_size], y_test[i * batch_size:(i + 1) * batch_size])
                 for i in range(num_processes)])
            
            for predicted, error in results:
                total_predicted += predicted
                total_error += error

        return total_predicted, total_testing, total_error / total_testing

    def test_network_batch(self, X_test_batch: np.ndarray, y_test_batch: np.ndarray):
        total_predicted = 0
        error = 0
        for i in range(X_test_batch.shape[0]):
            expected_result = y_test_batch[i]
            output_array = Mnist.preprocess_mnist_datapoint(X_test_batch[i])

            for layer in self.layers:
                output_array = layer.forward(output_array)

            expected_output_array = one_hot_encode(y_test_batch[i], self.num_classes)
            error += categorical_cross_entropy_loss(expected_output_array, output_array)

            prediction = get_result(output_array)
            if prediction == expected_result:
                total_predicted += 1

        return total_predicted, error
    
    def trainable_parameters_to_csv(self):
        """Write kernels, weights and biases to CSV files"""
        for index, layer in enumerate(self.layers):
            if isinstance(layer, ConvolutionalLayer):
                layer.kernels_biases_to_csv(f'../data/kernels_biases/layer_{index}/')
            elif isinstance(layer, DenseLayer):
                layer.weights_biases_to_csv(f'../data/weights_biases/layer_{index}/')

    def errors_accuracies_to_csv(self):
        """Write the errors and accuracies to CSV"""
        errors_accuracies_df = pd.DataFrame({
            'train_error':self.train_errors,
            'test_error':self.test_errors,
            'train_accuracy':self.train_accuracies,
            'test_accuracy':self.test_accuracies
        })
        filepath = Path('../data/errors_accuracies.csv')
        filepath.parent.mkdir(parents=True, exist_ok=True)
        errors_accuracies_df.to_csv(filepath, encoding='utf-8', index=False)

    def get_prediction_confidence(self, image: np.ndarray) -> tuple[int, float]:
        """Returns the label corresponding to the model's output,
        as well how confident the model is with its prediction;
        assumes the last layer's activation is the Softmax function.
        """
        output_array = image
        for layer in self.layers:
            output_array = layer.forward(output_array)

        prediction = get_result(output_array)
        confidence = output_array[prediction, 0]

        return prediction, confidence
    
def get_lenet_5_pretrained():
    lenet_5_pretrained = ConvolutionalNeuralNetwork(
        [
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
        ],
        num_classes=10
    )

    return lenet_5_pretrained 
    
if __name__ == '__main__':
    # Instantiate and train the CNN

    lenet_5 = ConvolutionalNeuralNetwork(
        [
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
        ],
        num_classes=10
    )


    print('mini_batch_size = 32, learning_rate = 0.07')
    start_time = time()
    lenet_5.fit(epochs=7, mini_batch_size=32, learning_rate=0.07, step=1, decay_rate=0.9)
    lenet_5.errors_accuracies_to_csv()
    lenet_5.trainable_parameters_to_csv()
    end_time = time()
    print(f'Time elapsed: {end_time - start_time} s')
    
    # write_to_csv = input('Write trained parameters to CSV? (y) ')
    # if write_to_csv == 'y':
    #     lenet_5.trainable_parameters_to_csv()
    #     print('Wrote trained parameters to CSV')

    # lenet_5_pretrained = ConvolutionalNeuralNetwork(
    #     [
    #         ConvolutionalLayer((1, 32, 32), (6, 28, 28), '../data/kernels_biases/layer_0'),   # 0
    #         ActivationLayer(tanh, derivative_tanh),
    #         AveragePoolingLayer(2),
    #         ConvolutionalLayer((6, 14, 14), (16, 10, 10), '../data/kernels_biases/layer_3'),  # 3
    #         ActivationLayer(tanh, derivative_tanh),
    #         AveragePoolingLayer(2),
    #         ConvolutionalLayer((16, 5, 5), (120, 1, 1), '../data/kernels_biases/layer_6'),    # 6
    #         ActivationLayer(tanh, derivative_tanh),
    #         ReshapeLayer((120, 1, 1), (120, 1)),
    #         DenseLayer(120, 84, '../data/weights_biases/layer_9'),                            # 9
    #         ActivationLayer(tanh, derivative_tanh),
    #         DenseLayer(84, 10, '../data/weights_biases/layer_11'),                            # 11
    #         SoftmaxLayer(),
    #     ],
    #     num_classes=10
    # )

    # print(lenet_5_pretrained.test_network_async())
