from dense_layer import DenseLayer
from activation_layer import ActivationLayer
from utils import tanh, derivative_tanh, mse, derivative_mse, load_mnist, one_hot_encode, get_result

mnist_train, mnist_test = load_mnist()

dense_neural_network = [
    DenseLayer(784, 40),
    ActivationLayer(tanh, derivative_tanh),
    DenseLayer(40, 10),
    ActivationLayer(tanh, derivative_tanh)
]

epochs = 20
learning_rate = 0.01

training_data = mnist_train.copy()
testing_data = mnist_test.copy()

total_training = training_data.shape[0]
total_testing = testing_data.shape[0]

for epoch in range(epochs):
    error = 0

    for i in range(total_training):
        expected_output_vector = one_hot_encode(training_data[i][0], 10)
        output_vector = training_data[i][1:] / 255
        output_vector = output_vector[..., None]

        # forward propagation
        for layer in dense_neural_network:
            output_vector = layer.forward(output_vector)

        error += mse(expected_output_vector, output_vector)
        
        # backwards propagation
        gradient = derivative_mse(expected_output_vector, output_vector)
        for layer in reversed(dense_neural_network):
            gradient = layer.backwards(gradient, learning_rate)

    error /= total_training
    print(f'Error: {error}')

    total_predicted = 0
    # test the network after each epoch
    for i in range(total_testing):
        expected_result = training_data[i][0]
        output_vector = training_data[i][1:] / 255
        output_vector = output_vector[..., None]

        # forward propagation
        for layer in dense_neural_network:
            output_vector = layer.forward(output_vector)
        
        prediction = get_result(output_vector)
        
        if prediction == expected_result:
            total_predicted += 1

    print(f'Epoch {epoch}: {total_predicted} / {total_testing}', end='\n\n')
