import numpy as np

# -------------------------------------------------
# Optimizers
# -------------------------------------------------

class SGD:

    def update(self, parameters, gradients, lr):

        for idx in range(len(parameters)):
            parameters[idx] = parameters[idx] - lr * gradients[idx]


# -------------------------------------------------
# Momentum Gradient Descent
# -------------------------------------------------

class Momentum:

    def __init__(self, beta=0.9):
        self.beta = beta
        self.velocity = None

    def update(self, parameters, gradients, lr):

        if self.velocity is None:
            self.velocity = [np.zeros_like(p) for p in parameters]

        for idx in range(len(parameters)):
            self.velocity[idx] = self.beta * self.velocity[idx] + (1 - self.beta) * gradients[idx]
            parameters[idx] -= lr * self.velocity[idx]


# -------------------------------------------------
# Nesterov Accelerated Gradient
# -------------------------------------------------

class Nesterov:

    def __init__(self, beta=0.9):
        self.beta = beta
        self.velocity = None

    def update(self, parameters, gradients, lr):

        if self.velocity is None:
            self.velocity = [np.zeros_like(p) for p in parameters]

        for idx in range(len(parameters)):

            previous_v = self.velocity[idx]

            self.velocity[idx] = self.beta * self.velocity[idx] - lr * gradients[idx]

            parameters[idx] += -self.beta * previous_v + (1 + self.beta) * self.velocity[idx]


# -------------------------------------------------
# AdaGrad
# -------------------------------------------------

class AdaGrad:

    def __init__(self, eps=1e-8):
        self.eps = eps
        self.cache = None

    def update(self, parameters, gradients, lr):

        if self.cache is None:
            self.cache = [np.zeros_like(p) for p in parameters]

        for idx in range(len(parameters)):

            self.cache[idx] += gradients[idx] ** 2

            adjusted_lr = lr / (np.sqrt(self.cache[idx]) + self.eps)

            parameters[idx] -= adjusted_lr * gradients[idx]


# -------------------------------------------------
# RMSProp
# -------------------------------------------------

class RMSProp:

    def __init__(self, beta=0.9, eps=1e-8):
        self.beta = beta
        self.eps = eps
        self.running_avg = None

    def update(self, parameters, gradients, lr):

        if self.running_avg is None:
            self.running_avg = [np.zeros_like(p) for p in parameters]

        for idx in range(len(parameters)):

            self.running_avg[idx] = (
                self.beta * self.running_avg[idx]
                + (1 - self.beta) * (gradients[idx] ** 2)
            )

            parameters[idx] -= lr * gradients[idx] / (np.sqrt(self.running_avg[idx]) + self.eps)


# -------------------------------------------------
# Muon Optimizer
# -------------------------------------------------

class Muon:

    def __init__(self, beta=0.9, eps=1e-8):
        self.beta = beta
        self.eps = eps
        self.momentum = None

    def update(self, parameters, gradients, lr):

        if self.momentum is None:
            self.momentum = [np.zeros_like(p) for p in parameters]

        for idx in range(len(parameters)):

            self.momentum[idx] = (
                self.beta * self.momentum[idx]
                + (1 - self.beta) * gradients[idx]
            )

            norm_val = np.linalg.norm(self.momentum[idx]) + self.eps

            parameters[idx] -= lr * self.momentum[idx] / norm_val


# -------------------------------------------------
# Adam Optimizer
# -------------------------------------------------

class Adam:

    def __init__(self, beta1=0.9, beta2=0.999, eps=1e-8):

        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.m = None
        self.v = None
        self.step = 0

    def update(self, parameters, gradients, lr):

        if self.m is None:
            self.m = [np.zeros_like(p) for p in parameters]
            self.v = [np.zeros_like(p) for p in parameters]

        self.step += 1

        for idx in range(len(parameters)):

            self.m[idx] = self.beta1 * self.m[idx] + (1 - self.beta1) * gradients[idx]

            self.v[idx] = self.beta2 * self.v[idx] + (1 - self.beta2) * (gradients[idx] ** 2)

            m_hat = self.m[idx] / (1 - self.beta1 ** self.step)
            v_hat = self.v[idx] / (1 - self.beta2 ** self.step)

            parameters[idx] -= lr * m_hat / (np.sqrt(v_hat) + self.eps)