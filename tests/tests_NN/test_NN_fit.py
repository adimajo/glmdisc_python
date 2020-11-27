#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pytest
import numpy as np
import pandas as pd
import sklearn as sk
import math
import random
import glmdisc
import tensorflow
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.callbacks import EarlyStopping


def test_kwargs():
    n = 100
    d = 2
    x, y, theta = glmdisc.Glmdisc.generate_data(n, d)
    model = glmdisc.Glmdisc(algorithm="NN", validation=False, test=False)

    model.fit(predictors_cont=x,
              predictors_qual=None,
              labels=y,
              plot=True,
              optimizer=Adagrad(),
              callbacks=EarlyStopping())
    assert isinstance(model.model_nn.optimizer, tensorflow.python.keras.optimizer_v2.adagrad.Adagrad)
    assert isinstance(model.callbacks[-1], tensorflow.python.keras.callbacks.EarlyStopping)


def test_args_fit():
    n = 100
    d = 2
    x, y, theta = glmdisc.Glmdisc.generate_data(n, d)
    model = glmdisc.Glmdisc(algorithm="NN", criterion="gini", test=False, validation=False)
    cuts = ([0, 0.333, 0.666, 1])
    xd = np.ndarray.copy(x)

    for i in range(d):
        xd[:, i] = pd.cut(x[:, i], bins=cuts, labels=[0, 1, 2])

    with pytest.raises(ValueError):
        model.fit(predictors_cont=x,
                  predictors_qual=None,
                  labels=[])

    with pytest.raises(ValueError):
        model.fit(predictors_cont="blabla",
                  predictors_qual=None,
                  labels=[])

    with pytest.raises(ValueError):
        model.fit(predictors_cont=None,
                  predictors_qual="blabla",
                  labels=[])

    with pytest.raises(ValueError):
        model.fit(predictors_cont=None,
                  predictors_qual=None,
                  labels=y)

    with pytest.raises(ValueError):
        model.fit(predictors_cont=x,
                  predictors_qual=None,
                  labels=y[0:50])

    with pytest.raises(ValueError):
        model.fit(predictors_cont=None,
                  predictors_qual=xd,
                  labels=y[0:50])


def test_iter():
    n = 100
    d = 2
    x, y, theta = glmdisc.Glmdisc.generate_data(n, d)

    with pytest.raises(ValueError):
        model = glmdisc.Glmdisc(algorithm="NN")
        model.fit(predictors_cont=x, predictors_qual=None, labels=y, iter=-12)

    with pytest.raises(ValueError):
        model = glmdisc.Glmdisc(algorithm="NN")
        model.fit(predictors_cont=x, predictors_qual=None, labels=y, iter=100000000)


def test_calculate_shape_continu():
    n = 100
    d = 2
    x, y, theta = glmdisc.Glmdisc.generate_data(n, d)
    model = glmdisc.Glmdisc(algorithm="NN")
    model.fit(predictors_cont=x, predictors_qual=None, labels=y, iter=11)
    continu_complete_case = model._calculate_shape()
    assert model.n == n
    assert model.d_cont == d
    assert model.d_qual == 0
    assert continu_complete_case.shape == (100, 2)
    assert continu_complete_case.all


def test_calculate_shape_categorical():
    n = 100
    d = 2
    x, y, theta = glmdisc.Glmdisc.generate_data(n, d)
    model = glmdisc.Glmdisc(algorithm="NN", m_start=3)
    cuts = ([0, 0.16, 0.333, 0.5, 0.666, 0.85, 1])
    xd = np.ndarray.copy(x)
    for i in range(d):
        xd[:, i] = pd.cut(x[:, i], bins=cuts, labels=[0, 1, 2, 3, 4, 5])
    model.fit(predictors_cont=None, predictors_qual=xd, labels=y, iter=11)
    continu_complete_case = model._calculate_shape()
    assert model.n == n
    assert model.d_cont == 0
    assert model.d_qual == d
    assert continu_complete_case is None


def test_nan():
    n = 100
    d = 2
    x, y, theta = glmdisc.Glmdisc.generate_data(n, d)
    x[0, 0] = np.nan
    x[90, 1] = np.nan
    model = glmdisc.Glmdisc(algorithm="NN", criterion="bic")
    model.fit(predictors_cont=x, predictors_qual=None, labels=y, iter=11)


def test_split():
    n = 100
    d = 2
    x, y, theta = glmdisc.Glmdisc.generate_data(n, d)
    model = glmdisc.Glmdisc(algorithm="NN")
    random.seed(1)
    np.random.seed(1)
    model.fit(predictors_cont=x, predictors_qual=None, labels=y, iter=11)
    training = model.train
    validating = model.validate
    testing = model.test_rows

    model = glmdisc.Glmdisc(algorithm="NN")
    random.seed(1)
    np.random.seed(1)
    model.fit(predictors_cont=x, predictors_qual=None, labels=y, iter=11)
    np.testing.assert_array_equal(training, model.train)
    np.testing.assert_array_equal(validating, model.validate)
    np.testing.assert_array_equal(testing, model.test_rows)
    assert len(model.train) > 0
    assert len(model.validate) > 0
    assert len(model.test_rows) > 0

    model = glmdisc.Glmdisc(algorithm="NN", validation=False)
    model.fit(predictors_cont=x, predictors_qual=None, labels=y, iter=11)
    assert len(model.train) > 0
    assert model.validate is None
    assert len(model.test_rows) > 0

    model = glmdisc.Glmdisc(algorithm="NN", test=False)
    model.fit(predictors_cont=x, predictors_qual=None, labels=y, iter=11)
    assert len(model.train) > 0
    assert len(model.validate) > 0
    assert model.test_rows is None

    model = glmdisc.Glmdisc(algorithm="NN", validation=False, test=False)
    model.fit(predictors_cont=x, predictors_qual=None, labels=y, iter=11)
    assert len(model.train) > 0
    assert model.validate is None
    assert model.test_rows is None


def test_not_fit():
    n = 100
    d = 2
    x, y, theta = glmdisc.Glmdisc.generate_data(n, d)
    model = glmdisc.Glmdisc(algorithm="NN", test=False, validation=False)
    for i in range(100):
        random.seed(i)
        np.random.seed(i)
        model.fit(predictors_cont=x, predictors_qual=None, labels=y, iter=11)
        try:
            model.check_is_fitted()
        except glmdisc.NotFittedError:
            with pytest.raises(glmdisc.NotFittedError):
                model.check_is_fitted()
            break
