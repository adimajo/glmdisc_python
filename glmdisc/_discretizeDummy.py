#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
discretizeDummy module for the glmdisc class.
"""


def discretizeDummy(self, predictors_cont, predictors_qual):
    """
    Discretizes new continuous and categorical features using a previously
    fitted glmdisc object as Dummy Variables usable with the best_reglog object.

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

    return self.best_encoder_emap.transform(
            self.discretize(predictors_cont, predictors_qual).astype(str))
