#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pytest
import numpy as np
import pandas as pd
import sklearn as sk
import math
import random
import glmdisc


def test_args_fit():
    n = 100
    d = 2
    x, y, theta = glmdisc.Glmdisc.generate_data(n, d)
    model = glmdisc.Glmdisc()
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


def test_calculate_shape_continu():
    n = 100
    d = 2
    x, y, theta = glmdisc.Glmdisc.generate_data(n, d)
    model = glmdisc.Glmdisc(iter=11)
    model.fit(predictors_cont=x, predictors_qual=None, labels=y)
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
    model = glmdisc.Glmdisc(iter=11)
    cuts = ([0, 0.333, 0.666, 1])
    xd = np.ndarray.copy(x)
    for i in range(d):
        xd[:, i] = pd.cut(x[:, i], bins=cuts, labels=[0, 1, 2])
    model.fit(predictors_cont=None, predictors_qual=xd, labels=y)
    continu_complete_case = model._calculate_shape()
    assert model.n == n
    assert model.d_cont == 0
    assert model.d_qual == d
    assert continu_complete_case is None

    model = glmdisc.Glmdisc(iter=11, m_start=2)
    model.fit(predictors_cont=None, predictors_qual=xd, labels=y)


def test_calculate_criterion():
    n = 100
    d = 2
    x, y, theta = glmdisc.Glmdisc.generate_data(n, d)
    model = glmdisc.Glmdisc(iter=11, criterion="bic")
    random.seed(1)
    np.random.seed(1)
    model.fit(predictors_cont=x, predictors_qual=None, labels=y)
    emap = np.resize(np.array([np.where(
        np.random.multinomial(1,
                              pvals=[0.33, 0.33, 0.34]))[0][0] + 1 for _ in range(n * d)]),
                     (n, d))

    current_encoder_emap = sk.preprocessing.OneHotEncoder()
    current_encoder_emap.fit(X=emap.astype(str))

    model_emap = sk.linear_model.LogisticRegression(solver='liblinear',
                                                    C=1e40,
                                                    tol=0.001,
                                                    max_iter=25,
                                                    warm_start=False)
    model_emap.fit(X=current_encoder_emap.transform(emap.astype(str)),
                   y=y)

    modele_bic = model._calculate_criterion(emap, model_emap, current_encoder_emap)
    assert modele_bic < 0

    model = glmdisc.Glmdisc(iter=11, criterion="aic")
    random.seed(1)
    np.random.seed(1)
    model.fit(predictors_cont=x, predictors_qual=None, labels=y)
    assert math.isclose(model._calculate_criterion(emap, model_emap, current_encoder_emap), modele_bic)

    model = glmdisc.Glmdisc(iter=11, validation=False)
    random.seed(1)
    np.random.seed(1)
    model.fit(predictors_cont=x, predictors_qual=None, labels=y)
    modele_bic = model._calculate_criterion(emap, model_emap, current_encoder_emap)

    model = glmdisc.Glmdisc(iter=11, criterion="aic", validation=False)
    random.seed(1)
    np.random.seed(1)
    model.fit(predictors_cont=x, predictors_qual=None, labels=y)
    assert math.isclose(model._calculate_criterion(emap,
                                                   model_emap,
                                                   current_encoder_emap), modele_bic + (
        math.log(model.n) - 2) * model_emap.coef_.shape[1])

    model = glmdisc.Glmdisc(iter=11, criterion="gini")
    model.fit(predictors_cont=x, predictors_qual=None, labels=y)
    assert 0 <= model._calculate_criterion(emap, model_emap, current_encoder_emap) <= 1

    model = glmdisc.Glmdisc(iter=11, criterion="gini", validation=False)
    model.fit(predictors_cont=x, predictors_qual=None, labels=y)
    assert 0 <= model._calculate_criterion(emap, model_emap, current_encoder_emap) <= 1


def test_nan():
    n = 100
    d = 2
    x, y, theta = glmdisc.Glmdisc.generate_data(n, d)
    x[0, 0] = np.nan
    x[90, 1] = np.nan
    model = glmdisc.Glmdisc(iter=11, criterion="bic")
    model.fit(predictors_cont=x, predictors_qual=None, labels=y)


def test_split():
    n = 100
    d = 2
    x, y, theta = glmdisc.Glmdisc.generate_data(n, d)
    model = glmdisc.Glmdisc(iter=11)
    random.seed(1)
    np.random.seed(1)
    model.fit(predictors_cont=x, predictors_qual=None, labels=y)
    training = model.train
    validating = model.validate
    testing = model.test_rows

    model = glmdisc.Glmdisc(iter=11)
    random.seed(1)
    np.random.seed(1)
    model.fit(predictors_cont=x, predictors_qual=None, labels=y)
    np.testing.assert_array_equal(training, model.train)
    np.testing.assert_array_equal(validating, model.validate)
    np.testing.assert_array_equal(testing, model.test_rows)
    assert len(model.train) > 0
    assert len(model.validate) > 0
    assert len(model.test_rows) > 0

    model = glmdisc.Glmdisc(iter=11, validation=False)
    model.fit(predictors_cont=x, predictors_qual=None, labels=y)
    assert len(model.train) > 0
    assert model.validate is None
    assert len(model.test_rows) > 0

    model = glmdisc.Glmdisc(iter=11, test=False)
    model.fit(predictors_cont=x, predictors_qual=None, labels=y)
    assert len(model.train) > 0
    assert len(model.validate) > 0
    assert model.test_rows is None

    model = glmdisc.Glmdisc(iter=11, validation=False, test=False)
    model.fit(predictors_cont=x, predictors_qual=None, labels=y)
    assert len(model.train) > 0
    assert model.validate is None
    assert model.test_rows is None


def test_not_fit():
    n = 100
    d = 2
    x, y, theta = glmdisc.Glmdisc.generate_data(n, d)
    model = glmdisc.Glmdisc(iter=11)
    for i in range(100):
        random.seed(i)
        np.random.seed(i)
        model.fit(predictors_cont=x, predictors_qual=None, labels=y)
        try:
            model.check_is_fitted()
        except glmdisc.NotFittedError:
            with pytest.raises(glmdisc.NotFittedError):
                model.check_is_fitted()
            break
