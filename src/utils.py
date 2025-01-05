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


def one_hot_encode(y: int, num_classes: int):
    one_hot_y = np.zeros((num_classes, 1))
    one_hot_y[y, 0] = 1
    return one_hot_y


def get_result(y_pred: np.ndarray):
    """Convert predicted output to one of the output classes;
    gets the line number of the max element

    :param y_pred: np.ndarray of shape (n, 1)
    :returns: int
    """
    max_element = y_pred[0, 0]
    max_element_index = 0
    for i in range(1, y_pred.shape[0]):
        if y_pred[i, 0] > max_element:
            max_element = y_pred[i, 0]
            max_element_index = i

    return max_element_index


def shuffle_data(X: np.ndarray, y: np.ndarray):
    indices = np.random.permutation(X.shape[0])

    return X[indices], y[indices]


def step_learning_rate_decay(learning_rate: float, epoch: int, step: int, decay_rate: float):
    """Method used to decrease the learning rate every step epochs"""
    return learning_rate * decay_rate ** ((epoch + 1) // step)


# ---------- cost functions and their derivatives ----------


def categorical_cross_entropy_loss(y_true: np.ndarray, y_pred: np.ndarray):
    epsilon = 1e-7  # Avoid numerical instability (NaN) by incorporating a small epsilon value
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.sum(y_true * np.log(y_pred))


def derivative_categorical_cross_entropy_loss(y_true: np.ndarray, y_pred: np.ndarray):
    epsilon = 1e-7  # Avoid numerical instability (NaN) by incorporating a small epsilon value
    return -y_true / (y_pred + epsilon)