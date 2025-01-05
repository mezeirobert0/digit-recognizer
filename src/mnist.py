import numpy as np
import pandas as pd

class Mnist:
    def __init__(self):
        mnist_train = pd.read_csv("../data/datasets/mnist_train.csv").to_numpy()
        mnist_test = pd.read_csv("../data/datasets/mnist_test.csv").to_numpy()

        self.y_train = mnist_train[:, 0]  # the label of every input is  on the first column
        self.y_test = mnist_test[:, 0]

        X_train = mnist_train[:, 1:]  # the input itself starts from the second column
        X_test = mnist_test[:, 1:]

        X_train_normalized = X_train / 255  # min value is 0, max value is 255; confine to range [0, 1]
        X_test_normalized = X_test / 255

        # The mean and standard deviation of the normalized training dataset
        # will be used for standardizing both the training and testing datasets,
        # as well as any image from the GUI
        self.X_train_mean = np.mean(X_train_normalized)
        self.X_train_std = np.std(X_train_normalized)

        # standardization: normalize
        self.X_train_standardized = (X_train_normalized - self.X_train_mean) / self.X_train_std
        self.X_test_standardized = (X_test_normalized - self.X_train_mean) / self.X_train_std

    def get_X_y_train(self):
        return self.X_train_standardized.copy(), self.y_train.copy()
    
    def get_X_y_test(self):
        return self.X_test_standardized.copy(), self.y_test.copy()

    def get_X_train_mean(self):
        return self.X_train_mean
    
    def get_X_train_std(self):
        return self.X_train_std
    
    @staticmethod
    def preprocess_mnist_datapoint(datapoint: np.ndarray):
        """Reshape to (28, 28), pad with 2 on each side, add extra dimension"""
        if datapoint.shape != (28, 28):
            return np.pad(datapoint.reshape(28, 28), 2)[None, ...]   # reshape, pad with 2 on each side, add extra dimension

        return np.pad(datapoint, 2)[None, ...]