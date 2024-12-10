from layer import Layer
from dense_layer import DenseLayer
from activation_layer import ActivationLayer
from utils import tanh, derivative_tanh, mse, derivative_mse, load_mnist, one_hot_encode, get_result
import numpy as np

mnist_train, mnist_test = load_mnist()

dense_neural_network = [
    DenseLayer(784, 64),
    ActivationLayer(tanh, derivative_tanh),
    DenseLayer(64, 10),
    ActivationLayer(tanh, derivative_tanh)
]

epochs = 10
mini_batch_size = 20
learning_rate = 1

training_data = mnist_train.copy()
testing_data = mnist_test.copy()

total_training = training_data.shape[0]
batches = total_training // mini_batch_size

total_testing = testing_data.shape[0]

for epoch in range(epochs):
    for batch in range(batches):
        for i in range(batch * mini_batch_size, (batch + 1) * mini_batch_size):
            expected_output_vector = one_hot_encode(mnist_train[i][0], 10)
            output_vector = training_data[i][1:]
            output_vector = output_vector[..., None]

            # forward propagation
            for layer in dense_neural_network:
                output_vector = layer.forward(output_vector)
            
            # backwards propagation
            gradient = derivative_mse(output_vector, expected_output_vector)
            for layer in reversed(dense_neural_network):
                gradient = layer.backwards(gradient, learning_rate)

    total_predicted = 0
    # test the network after each epoch
    for i in range(total_testing):
        expected_result = mnist_train[i][0]
        output_vector = mnist_train[i][1:]
        output_vector = output_vector[..., None]

        # forward propagation
        for layer in dense_neural_network:
            output_vector = layer.forward(output_vector)
        
        prediction = get_result(output_vector)
        
        if prediction == expected_result:
            total_predicted += 1

    print(f'Epoch {epoch}: {total_predicted} / {total_testing}')
