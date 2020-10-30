#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import random
import glmdisc


def test_discrete_data_cont():
    n = 500
    d = 2
    x, y, theta = glmdisc.Glmdisc.generate_data(n, d)
    model = glmdisc.Glmdisc(validation=False, test=False)
    random.seed(1)
    np.random.seed(1)
    model.fit(predictors_cont=x, predictors_qual=None, labels=y, iter=50)
    result = model.discrete_data()
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == 500


def test_discrete_data_qual():
    n = 500
    d = 2
    x, y, theta = glmdisc.Glmdisc.generate_data(n, d)
    cuts = ([0, 0.333, 0.666, 1])
    xd = np.ndarray.copy(x)

    for i in range(d):
        xd[:, i] = pd.cut(x[:, i], bins=cuts, labels=[0, 1, 2])

    model = glmdisc.Glmdisc(validation=False, test=False)
    random.seed(1)
    np.random.seed(1)
    model.fit(predictors_cont=None, predictors_qual=xd, labels=y, iter=50)
    result = model.discrete_data()
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == 500


def test_discrete_data_val_cont():
    n = 1000
    d = 2
    x, y, theta = glmdisc.Glmdisc.generate_data(n, d)
    model = glmdisc.Glmdisc(validation=True, test=False)
    random.seed(1)
    np.random.seed(1)
    model.fit(predictors_cont=x, predictors_qual=None, labels=y, iter=80)
    result = model.discrete_data()
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == 400


def test_discrete_data_val_qual():
    n = 500
    d = 2
    x, y, theta = glmdisc.Glmdisc.generate_data(n, d)
    cuts = ([0, 0.333, 0.666, 1])
    xd = np.ndarray.copy(x)

    for i in range(d):
        xd[:, i] = pd.cut(x[:, i], bins=cuts, labels=[0, 1, 2])

    model = glmdisc.Glmdisc(validation=True, test=False)
    random.seed(1)
    np.random.seed(1)
    model.fit(predictors_cont=None, predictors_qual=xd, labels=y, iter=50)
    result = model.discrete_data()
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == 200


def test_discrete_data_test_cont():
    n = 500
    d = 2
    x, y, theta = glmdisc.Glmdisc.generate_data(n, d)
    model = glmdisc.Glmdisc(validation=False, test=True)
    random.seed(1)
    np.random.seed(1)
    model.fit(predictors_cont=x, predictors_qual=None, labels=y, iter=50)
    result = model.discrete_data()
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == 200


def test_discrete_data_test_qual():
    n = 500
    d = 2
    x, y, theta = glmdisc.Glmdisc.generate_data(n, d)
    cuts = ([0, 0.333, 0.666, 1])
    xd = np.ndarray.copy(x)

    for i in range(d):
        xd[:, i] = pd.cut(x[:, i], bins=cuts, labels=[0, 1, 2])

    model = glmdisc.Glmdisc(validation=False, test=True)
    random.seed(1)
    np.random.seed(1)
    model.fit(predictors_cont=None, predictors_qual=xd, labels=y, iter=50)
    result = model.discrete_data()
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == 200


def test_discrete_data_both():
    n = 500
    d = 2
    x, y, theta = glmdisc.Glmdisc.generate_data(n, d)
    cuts = ([0, 0.333, 0.666, 1])
    xd = np.ndarray.copy(x)

    for i in range(d):
        xd[:, i] = pd.cut(x[:, i], bins=cuts, labels=[0, 1, 2])

    model = glmdisc.Glmdisc(validation=False, test=False)
    random.seed(1)
    np.random.seed(1)
    model.fit(predictors_cont=x, predictors_qual=xd, labels=y, iter=50)
    result = model.discrete_data()
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == 500


def test_discrete_data_val_both():
    n = 1000
    d = 2
    x, y, theta = glmdisc.Glmdisc.generate_data(n, d)
    cuts = ([0, 0.333, 0.666, 1])
    xd = np.ndarray.copy(x)

    for i in range(d):
        xd[:, i] = pd.cut(x[:, i], bins=cuts, labels=[0, 1, 2])

    model = glmdisc.Glmdisc(validation=True, test=False)
    random.seed(1)
    np.random.seed(1)
    model.fit(predictors_cont=x, predictors_qual=xd, labels=y, iter=100)
    result = model.discrete_data()
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == 400


def test_discrete_data_test_both():
    n = 500
    d = 2
    x, y, theta = glmdisc.Glmdisc.generate_data(n, d)
    cuts = ([0, 0.333, 0.666, 1])
    xd = np.ndarray.copy(x)

    for i in range(d):
        xd[:, i] = pd.cut(x[:, i], bins=cuts, labels=[0, 1, 2])

    model = glmdisc.Glmdisc(validation=False, test=True)
    random.seed(1)
    np.random.seed(1)
    model.fit(predictors_cont=x, predictors_qual=xd, labels=y, iter=50)
    result = model.discrete_data()
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == 200
