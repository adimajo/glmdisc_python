#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 16:22:17 2019

@author: adrien
"""


def predict(self, predictors_cont, predictors_qual):
    """Predicts the label values with new continuous and categorical features
    using a previously fitted glmdisc object.

    Keyword arguments:
    predictors_cont -- Continuous predictors to be discretized in a numpy
                        "numeric" array. Can be provided either here or with
                        the __init__ method.
    predictors_qual -- Categorical features which levels are to be merged
                        (also in a numpy "string" array). Can be provided
                        either here or with the __init__ method.
    """

    return self.best_reglog.predict_proba(
                self.discretizeDummy(predictors_cont, predictors_qual))
