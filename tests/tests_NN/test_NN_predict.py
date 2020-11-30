#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pytest
import glmdisc


def test_predict_new():
    n = 200
    d = 2
    x, y, theta = glmdisc.Glmdisc.generate_data(n, d)
    model = glmdisc.Glmdisc(algorithm="NN", validation=False, test=False)
    model.fit(predictors_cont=x[0:100], predictors_qual=None, labels=y[0:100], iter=11)
    results = model.predict(predictors_cont=x[100:200], predictors_qual=None)
    assert results.shape == (100, 2)
    assert (results > 0).all() and (results < 1).all()
    assert (results.sum(axis=1) == 1).all()

    model.fit(predictors_cont=x, predictors_qual=None, labels=y, iter=100)
    results = model.predict(predictors_cont=x, predictors_qual=None)

    glmdisc.gini(y, results[:, 1], sample_weight=None)
    glmdisc.gini(y, results[:, 1], sample_weight=np.ones(y.shape[0]))

    with pytest.raises(ValueError):
        glmdisc.gini(y, results[:, 1], level="toto")
