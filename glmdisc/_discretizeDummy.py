#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 16:21:57 2019

@author: adrien
"""


def discretizeDummy(self, predictors_cont,predictors_qual):
    """Discretizes new continuous and categorical features using a previously
    fitted glmdisc object as Dummy Variables usable with the best_reglog object.

    Keyword arguments:
    predictors_cont -- Continuous predictors to be discretized in a numpy
                        "numeric" array. Can be provided either here or with
                        the __init__ method.
    predictors_qual -- Categorical features which levels are to be merged
                        (also in a numpy "string" array). Can be provided
                        either here or with the __init__ method.
    """

    return self.best_encoder_emap.transform(
            self.discretize(predictors_cont, predictors_qual).astype(str))
