#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glmdisc


def test_plot():
    n = 500
    d = 2
    x, y, theta = glmdisc.Glmdisc.generate_data(n, d)
    cuts = ([0, 0.333, 0.666, 1])
    xd = np.ndarray.copy(x)
    for i in range(d):
        xd[:, i] = pd.cut(x[:, i], bins=cuts, labels=[0, 1, 2])

    model = glmdisc.Glmdisc(validation=False, test=False, iter=50)
    model.fit(predictors_cont=x, predictors_qual=xd, labels=y)
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

    model = glmdisc.Glmdisc(validation=False, test=False, iter=50)
    model.fit(predictors_cont=x, predictors_qual=xd, labels=y)
    model.plot(predictors_cont_number=0, predictors_qual_number=0)
    assert caplog.records[0].message == ("A single int (more than 0 and less than the "
                                         "number of columns in predictors_cont) must be "
                                         "provided for predictors_cont_number")
    assert caplog.records[1].message == ("A single int (more than 0 and less than the "
                                         "number of columns in predictors_qual) must be "
                                         "provided for predictors_qual_number")
