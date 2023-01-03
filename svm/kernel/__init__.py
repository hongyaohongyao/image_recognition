import numpy as np
import numpy.linalg as la


def linear(**kwargs):
    return lambda x, y: np.inner(x, y)


def poly(degree=3, coef0=0, **kwargs):
    return lambda x, y: (coef0 + np.inner(x, y))**degree


def sigmoid(gamma, coef0=0, **kwargs):
    return lambda x, y: np.tanh(gamma * np.dot(x, y) + coef0)


def rbf(gamma=10, **kwargs):
    return lambda x, y: np.exp(-gamma * la.norm(np.subtract(x, y)))