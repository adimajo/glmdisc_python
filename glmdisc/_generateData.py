#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""generate_data method for class glmdisc.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import combinations


@staticmethod
def generate_data(n, d, theta=None, plot=False):
    """
    Generates some toy continuous data that gets discretized, and a label
    is drawn from a logistic regression given the discretized features.

    :param int n:
        Number of observations to draw.
    :param int d:
        Number of features to draw.
    :param numpy.array theta:
        Logistic regression coefficient to use (if None, drawn from N(0,2)).
    :param bool plot:
        If true, plot the two first x axes with y as color.
    """
    cuts = ([0, 0.333, 0.666, 1])

    if theta is not None and not isinstance(theta, np.ndarray):
        raise ValueError("theta must be an np.array (or None).")
    elif theta is None:
        theta = np.array([[np.random.normal(0, 2) for _ in range(d)] for _ in range(len(cuts) - 1)])

    x = np.array(np.random.uniform(size=(n, d)))
    xd = np.ndarray.copy(x)

    for i in range(d):
        xd[:, i] = pd.cut(x[:, i], bins=cuts, labels=[0, 1, 2])

    log_odd = np.array([0] * n)
    for i in range(n):
        for j in range(d):
            log_odd[i] += theta[int(xd[i, j]), j]

    p = 1 / (1 + np.exp(- log_odd))
    y = np.random.binomial(1, p)

    if plot and d > 1:
        for unique_pair_of_feature_as_tuple in combinations(range(d), 2):
            plt.scatter(x[:, unique_pair_of_feature_as_tuple[0]],
                        x[:, unique_pair_of_feature_as_tuple[1]],
                        c=y,
                        s=[1 for _ in range(len(x))])

    return [x, y, theta]
