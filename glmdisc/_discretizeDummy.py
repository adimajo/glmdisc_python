#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""discretizeDummy module for the glmdisc class.
"""


def discretize_dummy(self, predictors_cont, predictors_qual):
    """
    Discretizes new continuous and categorical features using a previously
    fitted glmdisc object as Dummy Variables usable with the best_reglog object.

    :param numpy.array predictors_cont:
        Continuous predictors to be discretized in a numpy
        "numeric" array. Can be provided either here or with
        the __init__ method.
    :param numpy.array predictors_qual:
        Categorical features which levels are to be merged
        (also in a numpy "string" array). Can be provided
        either here or with the __init__ method.
    :returns: array of discretized features as dummy variables
    :rtype: numpy.array
    """

    return self.best_encoder_emap.transform(
        self.discretize(
            predictors_cont,
            predictors_qual).astype(int).astype(str))
