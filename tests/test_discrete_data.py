#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import scipy
import glmdisc


def test_discrete_data_cont():
    n = 500
    d = 2
    x, y, theta = glmdisc.Glmdisc.generate_data(n, d)
    model = glmdisc.Glmdisc(validation=False, test=False, iter=50)
    model.fit(predictors_cont=x, predictors_qual=None, labels=y)
    result = model.discrete_data()
    assert isinstance(result, scipy.sparse.csr.csr_matrix)
    assert result.shape[0] == 500


def test_discrete_data_qual():
    n = 500
    d = 2
    x, y, theta = glmdisc.Glmdisc.generate_data(n, d)
    cuts = ([0, 0.333, 0.666, 1])
    xd = np.ndarray.copy(x)

    for i in range(d):
        xd[:, i] = pd.cut(x[:, i], bins=cuts, labels=[0, 1, 2])

    model = glmdisc.Glmdisc(validation=False, test=False, iter=50)
    model.fit(predictors_cont=None, predictors_qual=xd, labels=y)
    result = model.discrete_data()
    assert isinstance(result, scipy.sparse.csr.csr_matrix)
    assert result.shape[0] == 500


def test_discrete_data_val_cont():
    n = 1000
    d = 2
    x, y, theta = glmdisc.Glmdisc.generate_data(n, d)
    model = glmdisc.Glmdisc(validation=True, test=False, iter=80)
    model.fit(predictors_cont=x, predictors_qual=None, labels=y)
    result = model.discrete_data()
    assert isinstance(result, scipy.sparse.csr.csr_matrix)
    assert result.shape[0] == 400


def test_discrete_data_val_qual():
    n = 500
    d = 2
    x, y, theta = glmdisc.Glmdisc.generate_data(n, d)
    cuts = ([0, 0.333, 0.666, 1])
    xd = np.ndarray.copy(x)

    for i in range(d):
        xd[:, i] = pd.cut(x[:, i], bins=cuts, labels=[0, 1, 2])

    model = glmdisc.Glmdisc(validation=True, test=False, iter=50)
    model.fit(predictors_cont=None, predictors_qual=xd, labels=y)
    result = model.discrete_data()
    assert isinstance(result, scipy.sparse.csr.csr_matrix)
    assert result.shape[0] == 200


def test_discrete_data_test_cont():
    n = 500
    d = 2
    x, y, theta = glmdisc.Glmdisc.generate_data(n, d)
    model = glmdisc.Glmdisc(validation=False, test=True, iter=50)
    model.fit(predictors_cont=x, predictors_qual=None, labels=y)
    result = model.discrete_data()
    assert isinstance(result, scipy.sparse.csr.csr_matrix)
    assert result.shape[0] == 200


def test_discrete_data_test_qual():
    n = 500
    d = 2
    x, y, theta = glmdisc.Glmdisc.generate_data(n, d)
    cuts = ([0, 0.333, 0.666, 1])
    xd = np.ndarray.copy(x)

    for i in range(d):
        xd[:, i] = pd.cut(x[:, i], bins=cuts, labels=[0, 1, 2])

    model = glmdisc.Glmdisc(validation=False, test=True, iter=50)
    model.fit(predictors_cont=None, predictors_qual=xd, labels=y)
    result = model.discrete_data()
    assert isinstance(result, scipy.sparse.csr.csr_matrix)
    assert result.shape[0] == 200


def test_discrete_data_both():
    n = 500
    d = 2
    x, y, theta = glmdisc.Glmdisc.generate_data(n, d)
    cuts = ([0, 0.333, 0.666, 1])
    xd = np.ndarray.copy(x)

    for i in range(d):
        xd[:, i] = pd.cut(x[:, i], bins=cuts, labels=[0, 1, 2])

    model = glmdisc.Glmdisc(validation=False, test=False, iter=50)
    model.fit(predictors_cont=x, predictors_qual=xd, labels=y)
    result = model.discrete_data()
    assert isinstance(result, scipy.sparse.csr.csr_matrix)
    assert result.shape[0] == 500


def test_discrete_data_val_both():
    n = 1000
    d = 2
    x, y, theta = glmdisc.Glmdisc.generate_data(n, d)
    cuts = ([0, 0.333, 0.666, 1])
    xd = np.ndarray.copy(x)

    for i in range(d):
        xd[:, i] = pd.cut(x[:, i], bins=cuts, labels=[0, 1, 2])

    model = glmdisc.Glmdisc(validation=True, test=False, iter=100)
    model.fit(predictors_cont=x, predictors_qual=xd, labels=y)
    result = model.discrete_data()
    assert isinstance(result, scipy.sparse.csr.csr_matrix)
    assert result.shape[0] == 400


def test_discrete_data_test_both():
    n = 500
    d = 2
    x, y, theta = glmdisc.Glmdisc.generate_data(n, d)
    cuts = ([0, 0.333, 0.666, 1])
    xd = np.ndarray.copy(x)

    for i in range(d):
        xd[:, i] = pd.cut(x[:, i], bins=cuts, labels=[0, 1, 2])

    model = glmdisc.Glmdisc(validation=False, test=True, iter=50)
    model.fit(predictors_cont=x, predictors_qual=xd, labels=y)
    result = model.discrete_data()
    assert isinstance(result, scipy.sparse.csr.csr_matrix)
    assert result.shape[0] == 200
