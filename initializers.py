"""
Initializer

This file contains various methods to initialize weights and biases.
"""

import numpy as np
import math


def xavier(input_dim, output_dim):
    """ std = sqrt(2)/sqrt(input_dim+output_dim)
    Xavier et al."""
    return np.random.normal(
        loc=0,
        scale=math.sqrt(2/(input_dim+output_dim)),
        size=(input_dim, output_dim)
    )


def he(input_dim, output_dim):
    """ std = sqrt(2)/sqrt(input_dim)
    He et al."""
    return np.random.normal(
        loc=0,
        scale=math.sqrt(2/input_dim),
        size=(input_dim, output_dim)
    )
