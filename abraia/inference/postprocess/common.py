"""Small numerical helpers used by output post-processing."""

import numpy as np


def normalize(img, mean, std):
    """Normalize an image using mean and standard deviation."""
    return ((img / 255 - np.array(mean)) / np.array(std)).astype(np.float32)


def sigmoid(x):
    """Compute the sigmoid activation function."""
    return 1 / (1 + np.exp(-x))


def softmax(x, axis=-1):
    """Compute softmax values along an axis."""
    values = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return values / values.sum(axis=axis, keepdims=True)


__all__ = ["normalize", "sigmoid", "softmax"]
