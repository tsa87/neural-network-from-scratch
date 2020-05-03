"""
Layer

The way to use this code is to stack the layers together in a list to build the
neural network. Methods operating on the Layer list can be found in Network.
"""

import numpy as np
from initializers import *


class Layer:
    """The abstract class for a single layer in a neural network architecture.
    You should subclass this and implement the method forward, backward and
    possibly __init__"""


    def __init__(self):
        """The constructor can initialize layer parameters"""
        pass


    def forward(self, input, production=False, weights=None):
        """ Returns output data of shape [batch, output dim] from input
        data of [batch, input dim].

        [production mode]: If off, layers will apply dropout when applicable
        [weights]: optional parameter to keep a weights list for regularization
        """
        return input    # a dummy layer


    def backward(self, input, grad_output, l2_a=0):
        """ Return the gradient, with respect to previous accumulated gradient,
        and the derivative of the current layer

        [l2_a]: L2 regularization penalization coefficient
        """

        dim = input.shape[1]
        d_layer_d_input = np.eye(dim)   #idenity matrix

        return grad_output @ d_layer_d_input


    def __str__(self):
        return "parent layer class"


#
# Activation Functions
#
class ReLU(Layer):
    """ Rectified Linear Unit is a non linear activation function
    f(x) = max(0, x)"""


    def forward(self, input, production=False, weights=None):
        return np.maximum(0, input)


    def backward(self, input, grad_output, l2_a=0):
        relu_grad = input > 0   # derivative=1 only applies to positive values
        return grad_output * relu_grad


    def __str__(self):
        return "ReLU class"


class Leaky_ReLU(Layer):
    """ Leaky ReLU is ReLU with a small slope added for negative values to address
    the dying ReLU problem.
    f(x) = max(ax, x)"""


    def __init__(self, slope=0.1):
        assert slope != 1   # slope cannot be 1, f(x) = max(x, x) = x is linear

        self.slope = slope


    def forward(self, input, production=False, weights=None):
        return np.maximum(self.slope * input, input)


    def backward(self, input, grad_output, l2_a=0):
        relu_grad = input > 0
        relu_grad[relu_grad == 0] = self.slope   # assign derivative to negative values

        return grad_output * relu_grad


    def __str__(self):
        return "Leaky ReLU class"


#
# Normalization Layers
#
class Batch_Norm(Layer):
    def __init__(self, variance=1, mean=0, alpha=0.4, learning_rate=0.1):
        self.variance = variance
        self.mean = mean

        self.learning_rate = learning_rate

        # for backward pass
        self.cache = None

        # Track variance and average of training batch for production mode.
        self.variance_sum = 0
        self.mean_sum = 0
        self.counter = 0

        self.alpha = alpha # avg = avg[training] * alpha + avg[sample] * (1 - alpha)


    #https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
    def forward(self, input, production=False, weights=None):
        #step1: calculate mean
        mean = np.mean(input, axis=0)

        if production:
            variance = np.std(input, axis=0)
            mean = self.mean_sum / counter * self.alpha + mean * (1 - self.alpha)
            variance = self.variance_sum / counter * self.alpha + variance * (1 - self.alpha)
            return variance
        else:
            # store the variance average
            self.mean_sum += mean
            self.variance_sum += variance
            self.counter += 1

        #step2: subtract mean vector of every trainings example
        diff = input - mean

        #step3: following the lower branch - calculation denominator
        diff_sqr = diff ** 2

        #step4: calculate variance
        var = 1./input.shape[0] * np.sum(diff, axis=0)

        #step5
        std = sqrt(variance)

        #step 6
        istd = 1. / std

        #step 7
        x_hat = diff * istd

        #step 8
        gxhat = self.variance * x_hat

        #step 9
        output = self.mean + gxhat

        self.cache = (x_hat, istd, diff, std, var, mean)

        return output


    def backward(self, input, grad_output, l2_a=0):
        assert cache is not None

        x_hat, istd, diff, std, var, mean = self.cache

        #step 9
        d_beta = np.sum(grad_output, axis=0)
        d_gxhat = grad_output

        #step 8
        d_gamma = np.sum(d_gxhat * x_hat, axis=0)
        d_xhat = d_gxhat * self.variance

        #step 7
        d_istd = np.sum(d_xhat * diff, axis=0)
        d_diff1 = d_xhat * istd

        #step 6
        d_std = -1 / (std ** 2) * d_istd

        #step 5
        d_var = 0.5 * 1. / sqrt(var) * d_std

        #step 4
        d_diff_squared = 1. / input.shape[0] * np.ones(input.shape) * d_var

        #step 3
        d_diff2 = 2 * diff * d_diff_squared

        #step 2
        d_x1 = d_diff1 + d_diff2
        d_mean = -1 * np.sum(d_x1, axis=0)

        #step 1
        d_x2 = 1. / input.shape[0] * np.ones(input.shape) * d_mean

        #step 0
        dx = dx1 + dx2

        self.variance -= self.learning_rate * d_gamma
        self.beta -= self.learning_rate * d_beta

        return dx


#
# Connection Layers
#
class Dense(Layer):
    """ Also known as the fully connected layer, contains the weights and biases.
    f(x) = xW + b"""

    def __init__(
        self,
        input_dim,
        output_dim,
        learning_rate=0.1,
        initializer=None,
        momentum=0,
        dropout_p=0
    ):

        assert momentum >= 0 and momentum < 1
        assert dropout_p >= 0 and dropout_p < 1

        # structural configuration
        self.input_dim = input_dim
        self.output_dim = output_dim

        # software configuration
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.dropout_p = dropout_p

        # layer content
        self.biases = np.zeros(output_dim)

        self.weights = np.random.normal(scale=0.01, size=(input_dim, output_dim))
        if initializer == "xavier":
            self.weights = xavier(input_dim, output_dim)
        elif initializer == "he":
            self.weights = he(input_dim, output_dim)

        self.grad_momentum = np.zeros_like(self.weights)
        self.dropout_mask = np.ones(input_dim, dtype=int) > 0

    def forward(self, input, production=False, weights=None):
        assert input.shape[1] == self.input_dim
        assert True not in np.isnan(input)

        if weights is not None:
            weights.append(self.weights)

        if not production:
            self.dropout_mask = np.random.uniform(low=0, high=1.0, size=self.input_dim) > self.dropout_p
            input = input * self.dropout_mask

        else:
            input = input * (1 - self.dropout_p)

        output = input @ self.weights + self.biases
        return output


    def backward(self, input, grad_output, l2_a=0):
        assert input.shape[1] == self.input_dim and grad_output.shape[1] == self.output_dim

        # Shape: [batch, output_dim] * [output_dim, input_dim] => [batch, input_dim]
        grad_input = grad_output @ self.weights.T

        # Shape: [input_dim, ?] * [?, output_dim] => [input_dim, output_dim]
        grad_weights = input.T @ grad_output + 2 * l2_a * self.weights
        grad_biases = np.sum(grad_output, axis=0)

        assert grad_weights.shape == self.weights.shape and grad_biases.shape == self.biases.shape

        self.grad_momentum = self.momentum * self.grad_momentum + self.learning_rate * grad_weights

        # gradient descent step
        self.weights -= self.grad_momentum
        self.biases -= self.learning_rate * grad_biases

        return grad_input


    def __str__(self):
        return "Dense layer class"
