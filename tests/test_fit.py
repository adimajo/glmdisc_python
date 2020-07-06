#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pytest
import numpy as np
import pandas as pd
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


def test_calculate_criterion():
    n = 100
    d = 2
    x, y, theta = glmdisc.Glmdisc.generate_data(n, d)
    model = glmdisc.Glmdisc(iter=11, criterion="bic")
    model.fit(predictors_cont=x, predictors_qual=None, labels=y)
    model._calculate_criterion(emap, model_emap, current_encoder_emap)

    model = glmdisc.Glmdisc(iter=11, criterion="aic")
    model.fit(predictors_cont=x, predictors_qual=None, labels=y)
    model._calculate_criterion(emap, model_emap, current_encoder_emap)

    model = glmdisc.Glmdisc(iter=11, criterion="gini")
    model.fit(predictors_cont=x, predictors_qual=None, labels=y)
    model._calculate_criterion(emap, model_emap, current_encoder_emap)


def test_init_disc():
    n = 100
    d = 2
    x, y, theta = glmdisc.Glmdisc.generate_data(n, d)
    model = glmdisc.Glmdisc(iter=11, criterion="bic")


def test_split():
    n = 100
    d = 2
    x, y, theta = glmdisc.Glmdisc.generate_data(n, d)
    model = glmdisc.Glmdisc(iter=11, criterion="bic")

