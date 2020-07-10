#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import glmdisc


def test_discretize():
    n = 100
    d = 2
    x, y, theta = glmdisc.Glmdisc.generate_data(n, d)
    model = glmdisc.Glmdisc(validation=False, test=False, iter=11)
    model.fit(predictors_cont=x, predictors_qual=None, labels=y)
    emap = model.discretize(predictors_cont=x, predictors_qual=None)
    model.best_encoder_emap.transform(emap.astype(int).astype(str))
