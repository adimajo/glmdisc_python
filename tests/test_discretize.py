#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pytest
import glmdisc


def test_discretize_new():
    n = 200
    d = 2
    x, y, theta = glmdisc.Glmdisc.generate_data(n, d)
    model = glmdisc.Glmdisc(validation=False, test=False, iter=11)
    model.fit(predictors_cont=x[0:100], predictors_qual=None, labels=y[0:100])
    emap = model.discretize(predictors_cont=x[100:200], predictors_qual=None)
    model.best_encoder_emap.transform(emap.astype(int).astype(str))


def test_discretize_cont():
    n = 100
    d = 2
    x, y, theta = glmdisc.Glmdisc.generate_data(n, d)
    model = glmdisc.Glmdisc(validation=False, test=False, iter=11)
    model.fit(predictors_cont=x, predictors_qual=None, labels=y)
    emap = model.discretize(predictors_cont=x, predictors_qual=None)
    model.best_encoder_emap.transform(emap.astype(int).astype(str))


def test_discretize_qual():
    n = 500
    d = 2
    x, y, theta = glmdisc.Glmdisc.generate_data(n, d)
    cuts = ([0, 0.333, 0.666, 1])
    xd = np.ndarray.copy(x)

    for i in range(d):
        xd[:, i] = pd.cut(x[:, i], bins=cuts, labels=[0, 1, 2])

    model = glmdisc.Glmdisc(validation=False, test=False, iter=50)
    model.fit(predictors_cont=None, predictors_qual=xd, labels=y)
    emap = model.discretize(predictors_cont=None, predictors_qual=xd)
    model.best_encoder_emap.transform(emap.astype(int).astype(str))


def test_discretize_wrong():
    n = 100
    d = 2
    x, y, theta = glmdisc.Glmdisc.generate_data(n, d)
    cuts = ([0, 0.333, 0.666, 1])
    xd = np.ndarray.copy(x)

    for i in range(d):
        xd[:, i] = pd.cut(x[:, i], bins=cuts, labels=[0, 1, 2])

    model = glmdisc.Glmdisc(validation=False, test=False, iter=11)
    model.fit(predictors_cont=x, predictors_qual=None, labels=y)
    with pytest.raises(ValueError):
        model.discretize(predictors_cont=None, predictors_qual=xd)
    with pytest.raises(ValueError):
        model.discretize(predictors_cont=x, predictors_qual=xd)
