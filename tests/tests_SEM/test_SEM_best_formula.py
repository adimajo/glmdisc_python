#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import glmdisc
from glmdisc._bestFormula import _best_formula_cont_sem


def test_best_formula(caplog):
    n = 200
    d = 2
    x, y, theta = glmdisc.Glmdisc.generate_data(n, d)
    cuts = ([0, 0.333, 0.666, 1])
    xd = np.ndarray.copy(x)

    for i in range(d):
        xd[:, i] = pd.cut(x[:, i], bins=cuts, labels=[0, 1, 2])

    model = glmdisc.Glmdisc(validation=False, test=False)
    model.fit(predictors_cont=x, predictors_qual=xd, labels=y, iter=11)
    formula = model.best_formula()
    assert isinstance(formula, list)
    assert len(formula) == 2 * d
    for j in range(2 * d):
        assert isinstance(formula[j], list)
    assert ("No cut-points found for continuous variable 0" in caplog.records[-4].message) | \
        ("Cut-points found for continuous variable 0" in caplog.records[-4].message)
    assert ("No cut-points found for continuous variable 1" in caplog.records[-3].message) | \
        ("Cut-points found for continuous variable 1" in caplog.records[-3].message)
    assert ("No regroupments made for categorical variable 0" in caplog.records[-2].message) | \
        ("Regroupments made for categorical variable 0" in caplog.records[-2].message)
    assert ("No regroupments made for categorical variable 1" in caplog.records[-1].message) | \
        ("Regroupments made for categorical variable 1" in caplog.records[-1].message)


def test_best_formula_no_cont():
    n = 200
    d = 2
    x, y, theta = glmdisc.Glmdisc.generate_data(n, d)
    cuts = ([0, 0.333, 0.666, 1])
    xd = np.ndarray.copy(x)

    for i in range(d):
        xd[:, i] = pd.cut(x[:, i], bins=cuts, labels=[0, 1, 2])

    model = glmdisc.Glmdisc(validation=False, test=False)
    model.fit(predictors_cont=None, predictors_qual=xd, labels=y, iter=11)
    model.best_formula()


class _Stub:
    d_cont = 2


def test_best_formula_cont_sem_missing_values():
    stub = _Stub()
    stub.predictors_cont = np.array([[1., np.nan],
                                     [2., np.nan],
                                     [3., np.nan],
                                     [4., np.nan]])
    emap_best = np.array([[0, 0], [0, 0], [1, 0], [1, 0]])
    assert _best_formula_cont_sem(stub, emap_best) == [[3.], []]


def test_best_formula_cont_sem_missing_level():
    stub = _Stub()
    stub.predictors_cont = np.array([[1., 5.],
                                     [2., 6.],
                                     [3., np.nan],
                                     [np.nan, 7.]])
    emap_best = np.array([[0, 0], [0, 0], [1, 1], [2, 1]])
    assert _best_formula_cont_sem(stub, emap_best) == [[3.], [7.]]
