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

    model = glmdisc.Glmdisc(algorithm="NN", validation=False, test=False)
    model.fit(predictors_cont=x, predictors_qual=xd, labels=y, iter=50)
    formula = model.best_formula()
    assert isinstance(formula, list)
    assert len(formula) == 2 * d
    for j in range(2 * d):
        assert isinstance(formula[j], list)
    assert "No cut-points found for continuous variable 0" in caplog.records[-4].message
    assert "No cut-points found for continuous variable 1" in caplog.records[-3].message
    # assert "No regroupments made for categorical variable 0" in caplog.records[-2].message
    # assert "No regroupments made for categorical variable 1" in caplog.records[-1].message
