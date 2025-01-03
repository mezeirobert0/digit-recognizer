import numpy as np
import pandas as pd

# ---------- activations and their derivatives ----------


def leaky_ReLU(z, alpha=0.05):
    return np.where(z > 0, z, alpha * z)


def derivative_leaky_ReLU(z, alpha=0.05):
    return np.where(z > 0, 1, alpha)


def ReLU(z):
    return np.maximum(z, 0)


def derivative_ReLU(z):
    return z > 0


def sigmoid(z):
    return 1 / 1 + np.exp(-z)


def derivative_sigmoid(z):
    s = sigmoid(z)
    return s * (1 - s)


def tanh(z):
    return np.tanh(z)


def derivative_tanh(z):
    return 1 - np.tanh(z) ** 2


# ---------- general utilities ----------


def one_hot_encode(y: int, num_classes):
    one_hot_y = np.zeros((num_classes, 1))
    one_hot_y[y, 0] = 1
    return one_hot_y


def get_result(y_pred: np.ndarray):
    max_element = y_pred[0, 0]
    max_element_index = 0
    for i in range(1, y_pred.shape[0]):
        if y_pred[i, 0] > max_element:
            max_element = y_pred[i, 0]
            max_element_index = i

    return max_element_index


def load_mnist():
    mnist_train_df = pd.read_csv("../data/datasets/mnist_train.csv")
    mnist_test_df = pd.read_csv("../data/datasets/mnist_test.csv")
    # return mnist_train_df.to_numpy(), mnist_test_df.to_numpy()
    return np.array(mnist_train_df.to_numpy()), np.array(mnist_test_df.to_numpy())


def valid_correlation(image: np.ndarray, kernel: np.ndarray):
    """Valid cross-correlation method, with a stride of 1"""
    image_height = image.shape[0]
    image_width = image.shape[1]
    kernel_height = kernel.shape[0]
    kernel_width = kernel.shape[1]
    result = np.zeros((image_height - kernel_height + 1, image_width - kernel_width + 1))
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            result[i, j] = np.dot(image[i:i+kernel_height, j:j+kernel_width].flatten(), kernel.flatten())

    return result


def full_correlation(image: np.ndarray, kernel: np.ndarray):
    """full cross-correlation method, with a stride of 1"""

    kernel_height = kernel.shape[0]
    kernel_width = kernel.shape[1]

    return valid_correlation(np.pad(image, ((kernel_height - 1, kernel_height - 1), (kernel_width - 1, kernel_width - 1))), kernel)


def valid_convolution(image: np.ndarray, kernel: np.ndarray):
    """Valid convolution method, with a stride of 1"""

    return valid_correlation(image, np.flip(kernel))  # kernel needs to be flipped horizontally and vertically for convolution


def full_convolution(image: np.ndarray, kernel: np.ndarray):
    """Full convolution method, with a stride of 1"""

    return full_correlation(image, np.flip(kernel))  # kernel needs to be flipped horizontally and vertically for convolution


def step_learning_rate_decay(learning_rate: float, epoch: int, step: int, decay_rate: float):
    return learning_rate * decay_rate ** ((epoch + 1) // step)


def preprocess_mnist_datapoint(datapoint: np.ndarray):
    processed_datapoint = datapoint / 255
    return np.pad(processed_datapoint.reshape(28, 28), 2)[None, ...]   # reshape, pad with 2 on each side, add extra dimension


# ---------- cost functions and their derivatives ----------


def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))


def derivative_mse(y_true, y_pred):
    return (2 / np.size(y_pred)) * (y_pred - y_true)


def categorical_cross_entropy_loss(y_true: np.ndarray, y_pred: np.ndarray):
    epsilon = 1e-7  # Avoid numerical instability (NaN) by incorporating a small epsilon value
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.sum(y_true * np.log(y_pred))


def derivative_categorical_cross_entropy_loss(y_true: np.ndarray, y_pred: np.ndarray):
    epsilon = 1e-7  # Avoid numerical instability (NaN) by incorporating a small epsilon value
    return -y_true / (y_pred + epsilon)