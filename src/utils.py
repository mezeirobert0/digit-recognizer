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

def one_hot_encode(y: int, num_classes):
    one_hot_y = np.zeros((num_classes, 1))
    one_hot_y[y][0] = 1
    return one_hot_y

def load_mnist():
    mnist_train_df = pd.read_csv("../data/datasets/mnist_train.csv")
    mnist_test_df = pd.read_csv("../data/datasets/mnist_test.csv")
    return mnist_train_df.to_numpy(), mnist_test_df.to_numpy()
