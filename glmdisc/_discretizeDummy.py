#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""discretizeDummy module for the glmdisc class.
"""
import numpy as np
import sklearn as sk


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
    if self.algorithm == "SEM":
        emap_dummy = self.best_encoder_emap.transform(
            self.discretize(
                predictors_cont,
                predictors_qual).astype(int).astype(str))
    else:
        results = self.discretize(predictors_cont, predictors_qual)
        emap_dummy = np.ones((predictors_cont.shape[0], 1))
        for j in range(self.d_cont + self.d_qual):
            results_dummy = sk.preprocessing.OneHotEncoder(categories='auto',
                                                           sparse=False,
                                                           handle_unknown="ignore").fit_transform(
                X=results[:, j].reshape(-1, 1))
            emap_dummy = np.concatenate(
                (emap_dummy,
                 results_dummy),
                axis=1)

    return emap_dummy
