import numpy as np


class WeightInit:

    @staticmethod
    def initialize_weights(layer_sizes, method="xavier"):

        weight_list = []
        bias_list = []

        num_layers = len(layer_sizes) - 1

        for layer in range(num_layers):

            input_size = layer_sizes[layer]
            output_size = layer_sizes[layer + 1]

            # Xavier Initialization
            if method == "xavier":
                scale = np.sqrt(1.0 / input_size)
                W = np.random.randn(input_size, output_size) * scale

            # He Initialization
            elif method == "he":
                scale = np.sqrt(2.0 / input_size)
                W = np.random.randn(input_size, output_size) * scale

            # Random small weights
            else:
                W = np.random.randn(input_size, output_size) * 0.01

            # Bias initialization
            b = np.zeros((1, output_size))

            weight_list.append(W)
            bias_list.append(b)

        return weight_list, bias_list