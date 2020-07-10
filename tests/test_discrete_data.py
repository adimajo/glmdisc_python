#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import scipy
import glmdisc


def test_discrete_data_cont():
    n = 100
    d = 2
    x, y, theta = glmdisc.Glmdisc.generate_data(n, d)
    model = glmdisc.Glmdisc(validation=False, test=False, iter=20)
    model.fit(predictors_cont=x, predictors_qual=None, labels=y)
    result = model.discrete_data()
    assert isinstance(result, scipy.sparse.csr.csr_matrix)
    assert result.shape[0] == 100


def test_discrete_data_qual():
    n = 100
    d = 2
    x, y, theta = glmdisc.Glmdisc.generate_data(n, d)
    cuts = ([0, 0.333, 0.666, 1])
    xd = np.ndarray.copy(x)

    for i in range(d):
        xd[:, i] = pd.cut(x[:, i], bins=cuts, labels=[0, 1, 2])

    model = glmdisc.Glmdisc(validation=False, test=False, iter=20)
    model.fit(predictors_cont=None, predictors_qual=xd, labels=y)
    result = model.discrete_data()
    assert isinstance(result, scipy.sparse.csr.csr_matrix)
    assert result.shape[0] == 100


def test_discrete_data_val_cont():
    n = 500
    d = 2
    x, y, theta = glmdisc.Glmdisc.generate_data(n, d)
    model = glmdisc.Glmdisc(validation=True, test=False, iter=60)
    model.fit(predictors_cont=x, predictors_qual=None, labels=y)
    result = model.discrete_data()
    assert isinstance(result, scipy.sparse.csr.csr_matrix)
    assert result.shape[0] == 200


def test_discrete_data_val_qual():
    n = 100
    d = 2
    x, y, theta = glmdisc.Glmdisc.generate_data(n, d)
    cuts = ([0, 0.333, 0.666, 1])
    xd = np.ndarray.copy(x)

    for i in range(d):
        xd[:, i] = pd.cut(x[:, i], bins=cuts, labels=[0, 1, 2])

    model = glmdisc.Glmdisc(validation=True, test=False, iter=20)
    model.fit(predictors_cont=None, predictors_qual=xd, labels=y)
    result = model.discrete_data()
    assert isinstance(result, scipy.sparse.csr.csr_matrix)
    assert result.shape[0] == 40


def test_discrete_data_test_cont():
    n = 100
    d = 2
    x, y, theta = glmdisc.Glmdisc.generate_data(n, d)
    model = glmdisc.Glmdisc(validation=False, test=True, iter=20)
    model.fit(predictors_cont=x, predictors_qual=None, labels=y)
    result = model.discrete_data()
    assert isinstance(result, scipy.sparse.csr.csr_matrix)
    assert result.shape[0] == 40


def test_discrete_data_test_qual():
    n = 100
    d = 2
    x, y, theta = glmdisc.Glmdisc.generate_data(n, d)
    cuts = ([0, 0.333, 0.666, 1])
    xd = np.ndarray.copy(x)

    for i in range(d):
        xd[:, i] = pd.cut(x[:, i], bins=cuts, labels=[0, 1, 2])

    model = glmdisc.Glmdisc(validation=False, test=True, iter=20)
    model.fit(predictors_cont=None, predictors_qual=xd, labels=y)
    result = model.discrete_data()
    assert isinstance(result, scipy.sparse.csr.csr_matrix)
    assert result.shape[0] == 40


def test_discrete_data_both():
    n = 100
    d = 2
    x, y, theta = glmdisc.Glmdisc.generate_data(n, d)
    cuts = ([0, 0.333, 0.666, 1])
    xd = np.ndarray.copy(x)

    for i in range(d):
        xd[:, i] = pd.cut(x[:, i], bins=cuts, labels=[0, 1, 2])

    model = glmdisc.Glmdisc(validation=False, test=False, iter=20)
    model.fit(predictors_cont=x, predictors_qual=xd, labels=y)
    result = model.discrete_data()
    assert isinstance(result, scipy.sparse.csr.csr_matrix)
    assert result.shape[0] == 100


def test_discrete_data_val_both():
    n = 300
    d = 2
    x, y, theta = glmdisc.Glmdisc.generate_data(n, d)
    cuts = ([0, 0.333, 0.666, 1])
    xd = np.ndarray.copy(x)

    for i in range(d):
        xd[:, i] = pd.cut(x[:, i], bins=cuts, labels=[0, 1, 2])

    model = glmdisc.Glmdisc(validation=True, test=False, iter=20)
    model.fit(predictors_cont=x, predictors_qual=xd, labels=y)
    result = model.discrete_data()
    assert isinstance(result, scipy.sparse.csr.csr_matrix)
    assert result.shape[0] == 120


def test_discrete_data_test_both():
    n = 100
    d = 2
    x, y, theta = glmdisc.Glmdisc.generate_data(n, d)
    cuts = ([0, 0.333, 0.666, 1])
    xd = np.ndarray.copy(x)

    for i in range(d):
        xd[:, i] = pd.cut(x[:, i], bins=cuts, labels=[0, 1, 2])

    model = glmdisc.Glmdisc(validation=False, test=True, iter=20)
    model.fit(predictors_cont=x, predictors_qual=None, labels=y)
    result = model.discrete_data()
    assert isinstance(result, scipy.sparse.csr.csr_matrix)
    assert result.shape[0] == 40
