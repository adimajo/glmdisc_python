#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glmdisc
import pytest


def test_plot():
    n = 500
    d = 2
    x, y, theta = glmdisc.Glmdisc.generate_data(n, d)
    cuts = ([0, 0.333, 0.666, 1])
    xd = np.ndarray.copy(x)
    for i in range(d):
        xd[:, i] = pd.cut(x[:, i], bins=cuts, labels=[0, 1, 2])

    model = glmdisc.Glmdisc(validation=False, test=False)
    model.fit(predictors_cont=x, predictors_qual=xd, labels=y, iter=50)
    model.plot()
    plt.close('all')

    model.plot(plot_type="logodd")
    plt.close('all')

    model.plot(predictors_cont_number=1, predictors_qual_number=1)
    plt.close('all')


def test_plot_log(caplog):
    n = 500
    d = 2
    x, y, theta = glmdisc.Glmdisc.generate_data(n, d)
    cuts = ([0, 0.333, 0.666, 1])
    xd = np.ndarray.copy(x)
    for i in range(d):
        xd[:, i] = pd.cut(x[:, i], bins=cuts, labels=[0, 1, 2])

    model = glmdisc.Glmdisc(validation=False, test=False)
    model.fit(predictors_cont=x, predictors_qual=xd, labels=y, iter=50)
    model.plot(predictors_cont_number=0, predictors_qual_number=0)
    assert caplog.records[-2].message != ("A single int (more than 0 and less than the "
                                          "number of columns in predictors_cont) must be "
                                          "provided for predictors_cont_number")
    assert caplog.records[-1].message != ("A single int (more than 0 and less than the "
                                          "number of columns in predictors_qual) must be "
                                          "provided for predictors_qual_number")
    with pytest.raises(ValueError):
        model.plot(predictors_cont_number=0, predictors_qual_number=-1)
    assert caplog.records[-1].message == ("A single int (more than 0 and less than the "
                                          "number of columns in predictors_qual) must be "
                                          "provided for predictors_qual_number")
    with pytest.raises(ValueError):
        model.plot(predictors_cont_number=-1, predictors_qual_number=0)
    assert caplog.records[-1].message == ("A single int (more than 0 and less than the "
                                          "number of columns in predictors_cont) must be "
                                          "provided for predictors_cont_number")


def _digitized(x):
    return np.digitize(x, [0.333, 0.666]).astype(float)


def test_plot_logodd_cont_only():
    np.random.seed(1)
    x, y, theta = glmdisc.Glmdisc.generate_data(300, 2)
    model = glmdisc.Glmdisc(validation=False, test=False)
    model.fit(predictors_cont=x, predictors_qual=None, labels=y, iter=11)
    model.plot(plot_type="logodd")
    plt.close('all')


def test_plot_logodd_qual_only():
    np.random.seed(1)
    x, y, theta = glmdisc.Glmdisc.generate_data(300, 2)
    model = glmdisc.Glmdisc(validation=False, test=False)
    model.fit(predictors_cont=None, predictors_qual=_digitized(x), labels=y, iter=11)
    model.plot(plot_type="logodd")
    plt.close('all')


def test_plot_logodd_single_feature():
    np.random.seed(1)
    x, y, theta = glmdisc.Glmdisc.generate_data(300, 2)
    model = glmdisc.Glmdisc(validation=False, test=False)
    model.fit(predictors_cont=x[:, :1], predictors_qual=None, labels=y, iter=11)
    model.plot(plot_type="logodd")
    plt.close('all')

    model = glmdisc.Glmdisc(validation=False, test=False)
    model.fit(predictors_cont=None, predictors_qual=_digitized(x[:, :1]), labels=y, iter=11)
    model.plot(plot_type="logodd")
    plt.close('all')
