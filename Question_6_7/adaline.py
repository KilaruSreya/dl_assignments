import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Adaline:
    
    def __init__(self, lr=1.0, epochs=1000):
        """
        Adaptive Linear Neuron model

        lr: learning rate controlling update size
        epochs: number of passes through training data
        """
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = 0.0


    def fit(self, X, y, X_valid=None, y_valid=None):
        """
        Train the ADALINE model using batch gradient descent.

        X : feature matrix (samples × features)
        y : target values
        X_valid, y_valid : optional validation dataset

        Returns:
            train_loss : list of training MSE values
            valid_loss : list of validation MSE values
        """

        samples, features = X.shape
        self.weights = np.zeros(features)
        self.bias = 0.0

        train_loss = []
        valid_loss = []

        for epoch in range(self.epochs):

            # Linear activation
            output = X @ self.weights + self.bias

            # Difference between true and predicted values
            residual = y - output

            # Parameter updates (Widrow–Hoff rule)
            self.weights += (self.lr / samples) * (X.T @ residual)
            self.bias += self.lr * residual.mean()

            # Training error
            mse_train = np.mean(residual ** 2)

            # Stop if training diverges
            if not np.isfinite(mse_train) or mse_train > 1e10:
                print("Training stopped due to divergence.")
                break

            train_loss.append(mse_train)

            # Validation error if validation data exists
            if X_valid is not None and y_valid is not None:
                pred_val = X_valid @ self.weights + self.bias
                mse_val = np.mean((y_valid - pred_val) ** 2)
                valid_loss.append(mse_val)

        return train_loss, valid_loss


    def predict(self, X):
        """Compute model output"""
        return X @ self.weights + self.bias


    def score(self, X, y):
        """Calculate mean squared error on given data"""
        predictions = self.predict(X)
        return np.mean((y - predictions) ** 2)