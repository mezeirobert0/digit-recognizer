import numpy as np
import pandas as pd


def leaky_ReLU(z, alpha=0.05):
    return np.where(z > 0, z, alpha * z)


def derivative_leaky_ReLU(z, alpha=0.05):
    return np.where(z > 0, 1, alpha)


def softmax(z):
    exp = np.exp(z)
    return exp / sum(exp)


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


def one_hot_encode(y: int, num_classes):
    one_hot_y = np.zeros((num_classes, 1))
    one_hot_y[y][0] = 1
    return one_hot_y


def get_result(y_pred: np.ndarray):
    max_element = y_pred[0][0]
    max_element_index = 0
    for i in range(1, y_pred.shape[0]):
        if y_pred[i][0] > max_element:
            max_element = y_pred[i][0]
            max_element_index = i

    return max_element_index




def load_mnist():
    mnist_train_df = pd.read_csv("../data/datasets/mnist_train.csv")
    mnist_test_df = pd.read_csv("../data/datasets/mnist_test.csv")
    return mnist_train_df.to_numpy(), mnist_test_df.to_numpy()


def cross_entropy_cost(y_pred, y_true):
    """
    Compute Cross-Entropy Loss.
    :param y_pred: np.ndarray, shape (m, 1), predicted probabilities (output of softmax)
    :param y_true: np.ndarray, shape (m, 1), one-hot encoded true labels
    :returns: float, cross-entropy loss
    """
    epsilon = 1e-12  # To avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    return -np.sum(y_true * np.log(y_pred)) / y_true.shape[1]


def mse(y_pred, y_true):
    return np.mean(np.power(y_true - y_pred, 2))


def derivative_mse(y_pred, y_true):
    return 2 / np.size(y_pred) * (y_pred - y_true)
