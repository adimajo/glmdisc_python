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

    log_odd = np.zeros(n)
    for i in range(n):
        for j in range(d):
            log_odd[i] += theta[int(xd[i, j]), j]

    p = 1 / (1 + np.exp(- log_odd))
    y = np.random.binomial(1, p, size=n)

    if plot and d > 1:
        for j1, j2 in combinations(range(d), 2):
            plt.figure()
            plt.scatter(x[:, j1], x[:, j2], c=y, s=1)
            plt.xlabel("x" + str(j1))
            plt.ylabel("x" + str(j2))
            plt.show(block=False)

    return [x, y, theta]
