import numpy as np


class Activations:

    # ------------------------------
    # Sigmoid
    # ------------------------------
    @staticmethod
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def sigmoid_derivative(z):
        sig = 1.0 / (1.0 + np.exp(-z))
        return sig * (1.0 - sig)


    # ------------------------------
    # Tanh
    # ------------------------------
    @staticmethod
    def tanh(z):
        return np.tanh(z)

    @staticmethod
    def tanh_derivative(z):
        t = np.tanh(z)
        return 1 - t * t


    # ------------------------------
    # ReLU
    # ------------------------------
    @staticmethod
    def relu(z):
        return np.maximum(z, 0)

    @staticmethod
    def relu_derivative(z):
        grad = np.zeros_like(z)
        grad[z > 0] = 1
        return grad


    # ------------------------------
    # Leaky ReLU
    # ------------------------------
    @staticmethod
    def leaky_relu(z, alpha=0.01):
        return np.where(z >= 0, z, alpha * z)

    @staticmethod
    def leaky_relu_derivative(z, alpha=0.01):
        grad = np.ones_like(z)
        grad[z < 0] = alpha
        return grad