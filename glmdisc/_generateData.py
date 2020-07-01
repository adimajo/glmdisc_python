#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_data method for class glmdisc.
"""


def generate_data(n, d, seed):
    """
    Generates some toy continuous data that gets discretized, and a label
    is drawn from a logistic regression given the discretized features.

    :param int n:
        Number of observations to draw.
    :param int d:
        Number of features to draw.
    :param int seed:
        The seed for random number generation.
    """

    random.seed(seed)
    x = np.array(np.random.uniform(size=(n, d)))
    xd = np.ndarray.copy(x)
    cuts = ([0, 0.333, 0.666, 1])

    for i in range(d):
        xd[:, i] = pd.cut(x[:, i], bins=cuts, labels=[0, 1, 2])

    theta = np.array([[1]*d]*(len(cuts)-1))
    theta[1, :] = 2
    theta[2, :] = -2

    log_odd = np.array([0]*n)
    for i in range(n):
        for j in range(d):
            log_odd[i] += theta[int(xd[i, j]), j]

    p = 1/(1+np.exp(-log_odd))
    y = np.random.binomial(1, p)

    return [x, y]
