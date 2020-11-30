#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pytest
import numpy as np
import pandas as pd
import random
import glmdisc
import tensorflow
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.callbacks import EarlyStopping


def test_kwargs(caplog):
    n = 100
    d = 2
    x, y, theta = glmdisc.Glmdisc.generate_data(n, d, plot=True)
    model = glmdisc.Glmdisc(algorithm="NN", validation=False, test=False)

    model.fit(predictors_cont=x,
              predictors_qual=None,
              labels=y,
              plot=True,
              optimizer=Adagrad(),
              callbacks=EarlyStopping())
    assert isinstance(model.model_nn['tensorflow_model'].optimizer,
                      tensorflow.python.keras.optimizer_v2.adagrad.Adagrad)
    assert isinstance(model.model_nn['callbacks'][-1],
                      tensorflow.python.keras.callbacks.EarlyStopping)
    with pytest.raises(ValueError):
        model.fit(predictors_cont=x,
                  predictors_qual=None,
                  labels=y,
                  plot="toto",
                  optimizer=Adagrad(),
                  callbacks=EarlyStopping())
    assert "plot parameter provided but not boolean" in caplog.records[-1].message


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
    training = model.train_rows
    validating = model.validation_rows
    testing = model.test_rows

    model = glmdisc.Glmdisc(algorithm="NN")
    random.seed(1)
    np.random.seed(1)
    model.fit(predictors_cont=x, predictors_qual=None, labels=y, iter=11)
    np.testing.assert_array_equal(training, model.train_rows)
    np.testing.assert_array_equal(validating, model.validation_rows)
    np.testing.assert_array_equal(testing, model.test_rows)
    assert len(model.train_rows) > 0
    assert len(model.validation_rows) > 0
    assert len(model.test_rows) > 0

    model = glmdisc.Glmdisc(algorithm="NN", validation=False)
    model.fit(predictors_cont=x, predictors_qual=None, labels=y, iter=11)
    assert len(model.train_rows) > 0
    assert model.validation_rows is None
    assert len(model.test_rows) > 0

    model = glmdisc.Glmdisc(algorithm="NN", test=False)
    model.fit(predictors_cont=x, predictors_qual=None, labels=y, iter=11)
    assert len(model.train_rows) > 0
    assert len(model.validation_rows) > 0
    assert model.test_rows is None

    model = glmdisc.Glmdisc(algorithm="NN", validation=False, test=False)
    model.fit(predictors_cont=x, predictors_qual=None, labels=y, iter=11)
    assert len(model.train_rows) > 0
    assert model.validation_rows is None
    assert model.test_rows is None


def test_not_fit():
    model = glmdisc.Glmdisc(algorithm="NN", test=False, validation=False)
    with pytest.raises(glmdisc.NotFittedError):
        model._check_is_fitted()
    n = 100
    d = 2
    x, y, theta = glmdisc.Glmdisc.generate_data(n, d)
    model = glmdisc.Glmdisc(algorithm="NN", test=False, validation=False, burn_in=20)
    with pytest.raises(glmdisc.NotFittedError):
        model.fit(predictors_cont=x, predictors_qual=None, labels=y, iter=11)


def test_fit():
    n = 100
    d = 2
    x, y, theta = glmdisc.Glmdisc.generate_data(n, d)
    model = glmdisc.Glmdisc(algorithm="NN", criterion="bic", validation=True)
    model.fit(predictors_cont=x, predictors_qual=None, labels=y, iter=20)
    model = glmdisc.Glmdisc(algorithm="NN", criterion="aic", validation=True)
    model.fit(predictors_cont=x, predictors_qual=None, labels=y, iter=20)
    model = glmdisc.Glmdisc(algorithm="NN", criterion="gini", validation=True)
    model.fit(predictors_cont=x, predictors_qual=None, labels=y, iter=20)
    model = glmdisc.Glmdisc(algorithm="NN", criterion="bic", validation=False)
    model.fit(predictors_cont=x, predictors_qual=None, labels=y, iter=20)
    model = glmdisc.Glmdisc(algorithm="NN", criterion="aic", validation=False)
    model.fit(predictors_cont=x, predictors_qual=None, labels=y, iter=20)
    model = glmdisc.Glmdisc(algorithm="NN", criterion="gini", validation=False)
    model.fit(predictors_cont=x, predictors_qual=None, labels=y, iter=20)
    model = glmdisc.Glmdisc(algorithm="NN", criterion="bic", test=False, validation=False)
    model.fit(predictors_cont=x, predictors_qual=None, labels=y, iter=20)
    model = glmdisc.Glmdisc(algorithm="NN", criterion="aic", test=False, validation=False)
    model.fit(predictors_cont=x, predictors_qual=None, labels=y, iter=20)
    model = glmdisc.Glmdisc(algorithm="NN", criterion="gini", test=False, validation=False)
    model.fit(predictors_cont=x, predictors_qual=None, labels=y, iter=20)
