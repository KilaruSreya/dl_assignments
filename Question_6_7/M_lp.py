import numpy as np
from activations import Activations
from losses import Loss
from optimizers import SGD, Momentum, Adam, Nesterov, AdaGrad, RMSProp, Muon
from weights import WeightInit


# -------------------------------------------------
# Multi Layer Perceptron
# -------------------------------------------------

class MLP:

    def __init__(self, layer_sizes, activations,
                 loss='cross_entropy', learning_rate=0.01,
                 optimizer='sgd', batch_size=32,
                 weight_init='xavier', regularization=None,
                 lambda_reg=0.01):

        self.layer_sizes = layer_sizes
        self.activation_list = activations
        self.lr = learning_rate
        self.batch_size = batch_size

        self.regularization = regularization
        self.lambda_reg = lambda_reg

        # initialize parameters
        self.weights, self.biases = WeightInit.initialize_weights(layer_sizes, weight_init)

        # select loss function
        if loss == "mse":
            self.loss_fn = Loss.mse
            self.loss_grad = Loss.mse_derivative
        elif loss == "mae":
            self.loss_fn = Loss.mae
            self.loss_grad = Loss.mae_derivative
        elif loss == "huber":
            self.loss_fn = Loss.huber
            self.loss_grad = Loss.huber_derivative
        else:
            self.loss_fn = Loss.mse
            self.loss_grad = Loss.mse_derivative

        # optimizer selection
        if optimizer == "momentum":
            self.optimizer = Momentum()
        elif optimizer == "adam":
            self.optimizer = Adam()
        elif optimizer == "nesterov":
            self.optimizer = Nesterov()
        elif optimizer == "adagrad":
            self.optimizer = AdaGrad()
        elif optimizer == "rmsprop":
            self.optimizer = RMSProp()
        elif optimizer == "muon":
            self.optimizer = Muon()
        else:
            self.optimizer = SGD()

    # -------------------------------------------------
    # Forward Propagation
    # -------------------------------------------------

    def forward(self, X):

        layer_inputs = []
        layer_outputs = [X]

        current = X

        for i in range(len(self.weights)):

            z = current @ self.weights[i] + self.biases[i]
            layer_inputs.append(z)

            act_name = self.activation_list[i]

            if act_name == "sigmoid":
                current = Activations.sigmoid(z)

            elif act_name == "tanh":
                current = Activations.tanh(z)

            elif act_name == "relu":
                current = Activations.relu(z)

            elif act_name == "leaky_relu":
                current = Activations.leaky_relu(z)

            elif act_name == "linear":
                current = z

            layer_outputs.append(current)

        return layer_outputs, layer_inputs

    # -------------------------------------------------
    # Backpropagation
    # -------------------------------------------------

    def backward(self, X, y):

        outputs, inputs = self.forward(X)

        grad_w = []
        grad_b = []

        error = self.loss_grad(y, outputs[-1])

        for layer in reversed(range(len(self.weights))):

            prev_activation = outputs[layer]

            dW = prev_activation.T @ error
            dB = np.sum(error, axis=0, keepdims=True)

            grad_w.insert(0, dW)
            grad_b.insert(0, dB)

            if layer > 0:

                z_prev = inputs[layer-1]
                act = self.activation_list[layer-1]

                if act == "sigmoid":
                    deriv = Activations.sigmoid_derivative(z_prev)

                elif act == "tanh":
                    deriv = Activations.tanh_derivative(z_prev)

                elif act == "relu":
                    deriv = Activations.relu_derivative(z_prev)

                elif act == "leaky_relu":
                    deriv = Activations.leaky_relu_derivative(z_prev)

                else:
                    deriv = 1

                error = (error @ self.weights[layer].T) * deriv

        # Regularization
        if self.regularization == "l1":
            for i in range(len(grad_w)):
                grad_w[i] += self.lambda_reg * np.sign(self.weights[i])

        elif self.regularization == "l2":
            for i in range(len(grad_w)):
                grad_w[i] += self.lambda_reg * self.weights[i]

        return grad_w, grad_b

    # -------------------------------------------------
    # Training Function
    # -------------------------------------------------

    def fit(self, X_train, y_train, X_val, y_val, epochs=100):

        history = {
            "train_loss": [],
            "val_loss": [],
            "update_mag": []
        }

        num_samples = X_train.shape[0]
        batch = self.batch_size if self.batch_size is not None else num_samples

        for ep in range(epochs):

            # shuffle data
            order = np.random.permutation(num_samples)
            X_train = X_train[order]
            y_train = y_train[order]

            epoch_updates = []

            for start in range(0, num_samples, batch):

                X_batch = X_train[start:start+batch]
                y_batch = y_train[start:start+batch]

                grad_w, grad_b = self.backward(X_batch, y_batch)

                parameters = self.weights + self.biases
                gradients = grad_w + grad_b

                old_params = [p.copy() for p in parameters]

                self.optimizer.update(parameters, gradients, self.lr)

                # compute update magnitude
                mag = 0
                for old, new in zip(old_params, parameters):
                    mag += np.linalg.norm(new - old)

                epoch_updates.append(mag)

            # compute losses
            train_out, _ = self.forward(X_train)
            val_out, _ = self.forward(X_val)

            train_loss = self.loss_fn(y_train, train_out[-1])
            val_loss = self.loss_fn(y_val, val_out[-1])

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["update_mag"].append(np.mean(epoch_updates))

        return history