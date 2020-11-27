#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""discretize method for glmdisc class.
"""
from collections import Counter

import numpy as np
import sklearn as sk
from loguru import logger
from scipy import stats

import glmdisc
from glmdisc._fitNN import _from_weights_to_proba_test


def _check_args_discretize_nn(self, predictors_cont, predictors_qual):
    if predictors_cont is not None:
        n_test = predictors_cont.shape[0]
    else:
        n_test = predictors_qual.shape[0]

    if predictors_cont is not None:
        d_1 = predictors_cont.shape[1]
    else:
        d_1 = 0

    if predictors_qual is not None:
        d_2 = predictors_qual.shape[1]
    else:
        d_2 = 0

    if d_1 != self.d_cont:
        msg = ('Shape of ' + str(d_1) +
               ' for predictors_cont does not match training set of size ' + str(self.d_cont) + '.')
        logger.error(msg)
        raise ValueError(msg)
    if d_2 != self.d_qual:
        msg = ('Shape of ' + str(d_2) +
               ' for predictors_cont does not match training set of size ' + str(self.d_qual) + '.')
        logger.error(msg)
        raise ValueError(msg)

    return n_test


def _check_args_discretize_sem(self, predictors_cont, predictors_qual):
    if predictors_cont is not None:
        n = predictors_cont.shape[0]
    else:
        n = predictors_qual.shape[0]

    if predictors_cont is not None:
        d_1 = predictors_cont.shape[1]
    else:
        d_1 = 0

    if predictors_qual is not None:
        d_2 = predictors_qual.shape[1]
    else:
        d_2 = 0

    d_1bis = [isinstance(x, sk.linear_model.LogisticRegression) for x in self.best_link]
    d_2bis = [isinstance(x, Counter) for x in self.best_link]

    if d_1 != sum(d_1bis) or d_1 != self.d_cont:
        msg = ('Shape of ' + str(d_1) +
               ' for predictors_cont does not match provided link function '
               'of size ' + str(sum(d_1bis)) + ' and/or training set of size ' + str(self.d_cont) + '.')
        logger.error(msg)
        raise ValueError(msg)
    if d_2 != sum(d_2bis) or d_2 != self.d_qual:
        msg = ('Shape of ' + str(d_2) +
               ' for predictors_cont does not match provided link function '
               'of size ' + str(sum(d_2bis)) + ' and/or training set of size ' + str(self.d_qual) + '.')
        logger.error(msg)
        raise ValueError(msg)

    return n, d_1, d_2, d_1bis, d_2bis


def _discretize_sem(self, predictors_cont, predictors_qual):
    """
    Discretizes new continuous and categorical features using a previously
    fitted glmdisc object.

    :param numpy.array predictors_cont:
        Continuous predictors to be discretized in a numpy
        "numeric" array. Can be provided either here or with
        the __init__ method.
    :param numpy.array predictors_qual:
        Categorical features which levels are to be merged
        (also in a numpy "string" array). Can be provided
        either here or with the __init__ method.
    """
    n, d_1, d_2, d_1bis, d_2bis = _check_args_discretize_sem(self, predictors_cont, predictors_qual)

    emap = np.zeros((n, d_1 + d_2))

    for j in range(d_1 + d_2):
        if d_1bis[j]:
            emap[np.invert(np.isnan(predictors_cont[:, j])), j] = np.argmax(
                self.best_link[j].predict_proba(predictors_cont[np.invert(np.isnan(predictors_cont[:, j])),
                                                                j].reshape(-1, 1)), axis=1)
            emap[np.isnan(predictors_cont[:, j]), j] = stats.describe(emap[:, j]).minmax[1] + 1
        elif d_2bis[j]:
            m = max(self.best_link[j].keys(), key=lambda key: key[1])[1]
            t = np.zeros((n, int(m) + 1))

            for i in range(n):
                for k in range(int(m) + 1):
                    t[i, k] = self.best_link[j][(int((self.affectations[j].transform(
                        np.ravel(predictors_qual[i, j - d_1])))), k)] / n

            emap[:, j] = np.argmax(t, axis=1)

        else:  # pragma: no cover
            msg = 'Loophole: please open an issue at https://github.com/adimajo/glmdisc_python/issues'
            logger.error(msg)
            raise ValueError(msg)

    return emap


def _discretize_nn(self, predictors_cont, predictors_qual):
    """
    Discretizes new continuous and categorical features using a previously
    fitted glmdisc object.

    :param numpy.array predictors_cont:
        Continuous predictors to be discretized in a numpy
        "numeric" array. Can be provided either here or with
        the __init__ method.
    :param numpy.array predictors_qual:
        Categorical features which levels are to be merged
        (also in a numpy "string" array). Can be provided
        either here or with the __init__ method.
    """
    if predictors_qual is not None:
        predictors_trans = np.zeros((predictors_qual.shape[0], self.d_qual))
        predictors_qual_dummy = []

        for j in range(self.d_qual):
            # Label encoding of qualitative input
            predictors_trans[:, j] = (self.affectations[j + self.d_cont].transform(
                predictors_qual[:, j])).astype(int)
            predictors_qual_dummy.append(np.squeeze(np.asarray(
                self.model_nn["one_hot_encoders_nn"][j].transform(predictors_trans[:, j].reshape(-1, 1)).todense())))
    else:
        predictors_trans = None

    n_test = _check_args_discretize_nn(self,
                                       predictors_cont,
                                       predictors_qual)

    proba = _from_weights_to_proba_test(self.d_cont,
                                        self.d_qual,
                                        [self.m_start] * self.d_cont,
                                        self.model_nn["callbacks"][1],
                                        predictors_cont,
                                        predictors_trans,
                                        n_test)

    results = [None] * (self.d_cont + self.d_qual)

    for j in range(self.d_cont + self.d_qual):
        results[j] = np.argmax(proba[j], axis=1)

    return np.vstack(results).T


def discretize(self, predictors_cont, predictors_qual):
    """
    Discretizes new continuous and categorical features using a previously
    fitted glmdisc object.

    :param numpy.array predictors_cont:
        Continuous predictors to be discretized in a numpy
        "numeric" array. Can be provided either here or with
        the __init__ method.
    :param numpy.array predictors_qual:
        Categorical features which levels are to be merged
        (also in a numpy "string" array). Can be provided
        either here or with the __init__ method.
    """

    self._check_is_fitted()

    glmdisc._fit._check_args(predictors_cont=predictors_cont,
                             predictors_qual=predictors_qual,
                             labels=None,
                             check_labels=False)

    if self.algorithm == "SEM":
        return _discretize_sem(self, predictors_cont, predictors_qual)
    else:
        return _discretize_nn(self, predictors_cont, predictors_qual)
