import numpy as np


class Loss:

    # ----------------------------------
    # Mean Squared Error (MSE)
    # ----------------------------------
    @staticmethod
    def mse(target, prediction):
        diff = target - prediction
        return np.mean(diff ** 2)

    @staticmethod
    def mse_derivative(target, prediction):
        n = target.shape[0]
        return (2.0 / n) * (prediction - target)


    # ----------------------------------
    # Mean Absolute Error (MAE)
    # ----------------------------------
    @staticmethod
    def mae(target, prediction):
        diff = target - prediction
        return np.mean(np.abs(diff))

    @staticmethod
    def mae_derivative(target, prediction):
        return np.sign(prediction - target)


    # ----------------------------------
    # Huber Loss
    # ----------------------------------
    @staticmethod
    def huber(target, prediction, delta=1.0):

        error = target - prediction
        abs_error = np.abs(error)

        quadratic = 0.5 * (error ** 2)
        linear = delta * (abs_error - 0.5 * delta)

        loss = np.where(abs_error <= delta, quadratic, linear)

        return np.mean(loss)

    @staticmethod
    def huber_derivative(target, prediction, delta=1.0):

        error = prediction - target
        abs_error = np.abs(error)

        grad_small = error
        grad_large = delta * np.sign(error)

        return np.where(abs_error <= delta, grad_small, grad_large)