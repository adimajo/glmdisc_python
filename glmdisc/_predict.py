#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""predict method for class glmdisc
"""


def predict(self, predictors_cont, predictors_qual):
    """
    Predicts the label values with new continuous and categorical features
    using a previously fitted glmdisc object.

    :param numpy.array predictors_cont:
        Continuous predictors to be discretized in a numpy
        "numeric" array. Can be provided either here or with
        the :code:`__init__` method.

    :param numpy.array predictors_qual:
        Categorical features which levels are to be merged
        (also in a numpy "string" array). Can be provided
        either here or with the :code:`__init__` method.
    """

    return self.best_reglog.predict_proba(self.discretize_dummy(predictors_cont, predictors_qual))
