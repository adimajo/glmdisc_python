#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import scipy
import glmdisc


def test_discrete_data():
    n = 100
    d = 2
    x, y, theta = glmdisc.Glmdisc.generate_data(n, d)
    model = glmdisc.Glmdisc(validation=False, test=False, iter=11)
    model.fit(predictors_cont=x, predictors_qual=None, labels=y)
    result = model.discrete_data()
    assert isinstance(result, scipy.sparse.csr.csr_matrix)
    assert result.shape[0] == 100
