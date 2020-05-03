"""
Network

This code provides methods to interact with the neural network, represented by
a list of layers.
"""


import numpy as np
from functions import *

def forward(network, X, production=False, weights=None):
    """ Pass the input X through layers and return the final activation vector.
    If production mode is off, layers will apply dropout methods if configured.
    """
    activations = [X]

    for layer in network:
        activation = layer.forward(activations[-1], production=production, weights=weights)
        activations.append(activation)

    assert len(activations) == len(network)+1
    return activations


def predict(network, X):
    logits = forward(network, X, production=True)[-1]
    return np.argmax(logits, axis=-1)


def train(network, X, y, weights=None, l2_a=0):

    activations = forward(network, X, production=False, weights=weights)
    logits = activations[-1]

    cross_entropy_loss = np.mean(softmax_logits(logits, y))
    L2_loss = np.sum(np.sum(np.square(W)) for W in weights) * l2_a / y.shape[0]
    loss = cross_entropy_loss + L2_loss
    loss_grad = grad_softmax_logits(logits, y)

    for i in range(len(network)-1, -1, -1):
        loss_grad = network[i].backward(activations[i], loss_grad, l2_a=l2_a)

    return loss
