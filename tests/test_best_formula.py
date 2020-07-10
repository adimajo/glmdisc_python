#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import glmdisc


def test_best_formula(caplog):
    n = 200
    d = 2
    x, y, theta = glmdisc.Glmdisc.generate_data(n, d)
    cuts = ([0, 0.333, 0.666, 1])
    xd = np.ndarray.copy(x)

    for i in range(d):
        xd[:, i] = pd.cut(x[:, i], bins=cuts, labels=[0, 1, 2])

    model = glmdisc.Glmdisc(validation=False, test=False, iter=11)
    model.fit(predictors_cont=x, predictors_qual=xd, labels=y)
    formula = model.best_formula()
    assert isinstance(formula, list)
    assert len(formula) == 2 * d
    for j in range(2 * d):
        assert isinstance(formula[j], list)
    assert len(caplog.records) == 4
    assert "Cut-points found for continuous variable" in caplog.records[0].message
    assert "Cut-points found for continuous variable" in caplog.records[1].message
    assert "Regroupments made for categorical variable" in caplog.records[2].message
    assert "Regroupments made for categorical variable" in caplog.records[3].message
