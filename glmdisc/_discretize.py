#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 16:21:31 2019

@author: adrien
"""


def discretize(self, predictors_cont, predictors_qual):
    """Discretizes new continuous and categorical features using a previously
    fitted glmdisc object.

    Keyword arguments:
    predictors_cont -- Continuous predictors to be discretized in a numpy
                        "numeric" array. Can be provided either here or with
                        the __init__ method.
    predictors_qual -- Categorical features which levels are to be merged
                        (also in a numpy "string" array). Can be provided
                        either here or with the __init__ method.
    """

    try:
        n = predictors_cont.shape[0]
    except AttributeError:
        n = predictors_qual.shape[0]

    try:
        d_1 = predictors_cont.shape[1]
    except AttributeError:
        d_1 = 0

    try:
        d_2 = predictors_qual.shape[1]
    except AttributeError:
        d_2 = 0

    d_1bis = [isinstance(x, sk.linear_model.logistic.LogisticRegression) for x in self.best_link]
    d_2bis = [isinstance(x, Counter) for x in self.best_link]

    if d_1 != sum(d_1bis): raise ValueError('Shape of predictors1 does not match provided link function')
    if d_2 != sum(d_2bis): raise ValueError('Shape of predictors2 does not match provided link function')

    emap = np.array([0] * n * (d_1 + d_2)).reshape(n, d_1 + d_2)

    for j in range(d_1 + d_2):
        if d_1bis[j]:
            emap[np.invert(np.isnan(predictors_cont[:, j])), j] = np.argmax(
                    self.best_link[j].predict_proba(
                            predictors_cont[np.invert(
                                    np.isnan(predictors_cont[:, j])), j].reshape(-1, 1)), axis=1)
            emap[np.isnan(predictors_cont[:, j]), j] = stats.describe(emap[:, j]).minmax[1] + 1
        elif d_2bis[j]:
            m = max(self.best_link[j].keys(), key=lambda key: key[1])[1]
            t = np.zeros((n, int(m) + 1))

            for l in range(n):
                for k in range(int(m) + 1):
                    t[l, k] = self.best_link[j][(int((self.affectations[j].transform(np.ravel(predictors_qual[l, j - d_1])))), k)] / n

            emap[:, j] = np.argmax(t, axis=1)

        else: raise ValueError('Not quantitative nor qualitative?')

    return emap
