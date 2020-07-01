#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.. module:: predict
    :synopsis: predict method for class glmdisc
"""


def predict(self, predictors_cont, predictors_qual):
    """
    Predicts the label values with new continuous and categorical features
    using a previously fitted glmdisc object.

    Parameters
    ----------
    predictors_cont : numpy.array
        Continuous predictors to be discretized in a numpy
        "numeric" array. Can be provided either here or with
        the __init__ method.
    predictors_qual : numpy.array
        Categorical features which levels are to be merged
        (also in a numpy "string" array). Can be provided
        either here or with the __init__ method.
    """

    return self.best_reglog.predict_proba(
                self.discretizeDummy(predictors_cont, predictors_qual))
