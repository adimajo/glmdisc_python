#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fit method for the glmdisc class.
"""
import numpy as np
import sklearn as sk
from loguru import logger
from scipy import stats

from glmdisc._fitNN import _fit_nn
from glmdisc._fitSEM import _fit_sem

NUMPY_NDARRAY_INPUTS = 'glmdisc only supports numpy.ndarray inputs'


def _check_args(predictors_cont, predictors_qual, labels, check_labels=True):
    """
    Checks inputs

    :param numpy.ndarray predictors_cont: continuous predictors
    :type predictors_cont: numpy.ndarray
    :param numpy.ndarray predictors_qual: categorical predictors
    :type predictors_qual: numpy.ndarray
    :param labels: binary labels
    :type labels: numpy.ndarray
    """
    # Test if predictors_cont is provided and if it's a numpy array
    raise_ndarray = False
    if predictors_cont is not None and not isinstance(predictors_cont, np.ndarray):
        raise_ndarray = True
    # Test if predictors_qual is provided and if it's a numpy array
    if predictors_qual is not None and not isinstance(predictors_qual, np.ndarray):
        raise_ndarray = True
    # Test if labels is provided and if it's a numpy array
    if check_labels and not isinstance(labels, np.ndarray):
        raise_ndarray = True
    if raise_ndarray:
        logger.error(NUMPY_NDARRAY_INPUTS)
        raise ValueError(NUMPY_NDARRAY_INPUTS)

    # Test if at least one of qual or cont is provided
    if predictors_cont is None and predictors_qual is None:
        msg = ('You must provide either qualitative or quantitative '
               'features')
        logger.error(msg)
        raise ValueError(msg)

    # Test if labels and predictors have same number of samples
    if check_labels and ((predictors_cont is not None and predictors_cont.shape[0] != labels.shape[0]) or
                         (predictors_qual is not None and predictors_qual.shape[0] != labels.shape[0])):
        msg = 'Predictors and labels must be of same size'
        logger.error(msg)
        raise ValueError(msg)


def _calculate_shape(self):
    """
    Calculates shape of inputs, stores number of samples and number of continuous and
    categorical predictors in self.

    :returns: array of positions of non np.nan continuous predictors
    :rtype: numpy.array
    """
    # Calculate shape of predictors (re-used multiple times)
    self.n = self.labels.shape[0]
    if self.predictors_cont is not None:
        self.d_cont = self.predictors_cont.shape[1]
    else:
        self.d_cont = 0

    if self.predictors_qual is not None:
        self.d_qual = self.predictors_qual.shape[1]
    else:
        self.d_qual = 0

    # Store location of missing continuous predictors; treat them as a separate level
    if self.predictors_cont is not None:
        continu_complete_case = np.invert(np.isnan(self.predictors_cont))
    else:
        continu_complete_case = None
    return continu_complete_case


def _init_disc(self, continu_complete_case):
    """
    Initializes :code:`affectations`, i.e. the list of label encoders for categorical features,
    chooses random values for the initial quantization randomly given :code:`m_start` if, for
    categorical features, :code:`m_start` is less than the number of levels. Otherwise, decrease by one.

    Parameters
    ----------
    continu_complete_case: array of missing inputs in contiuous features.

    Returns
    -------
    initial quantization and transformation of categorical features to integers
    """
    self.affectations = [None] * (self.d_cont + self.d_qual)
    edisc = np.random.choice(list(range(self.m_start)), size=(self.n, self.d_cont + self.d_qual))

    for j in range(self.d_cont):
        edisc[np.invert(continu_complete_case[:, j]), j] = self.m_start

    predictors_trans = np.zeros((self.n, self.d_qual))

    for j in range(self.d_qual):
        self.affectations[j + self.d_cont] = sk.preprocessing.LabelEncoder().fit(self.predictors_qual[:, j])
        if (self.m_start > stats.describe(sk.preprocessing.LabelEncoder().fit(self.predictors_qual[:, j]).transform(
                self.predictors_qual[:, j])).minmax[1] + 1):
            edisc[:, j + self.d_cont] = np.random.choice(list(range(
                stats.describe(sk.preprocessing.LabelEncoder().fit(
                    self.predictors_qual[:, j]).transform(
                        self.predictors_qual[:, j])).minmax[1])),
                size=self.n)
        else:
            edisc[:, j + self.d_cont] = np.random.choice(list(range(self.m_start)),
                                                         size=self.n)

        predictors_trans[:, j] = (self.affectations[j + self.d_cont].transform(
            self.predictors_qual[:, j])).astype(int)
    return edisc, predictors_trans


def _split(self):
    """
    Splits the dataset in train, validation and test given user-chosen parameters.
    """
    if self.validation and self.test:
        self.train_rows, self.validation_rows, self.test_rows = np.split(np.random.choice(self.n,
                                                                                          self.n,
                                                                                          replace=False),
                                                                         [int(.6 * self.n), int(.8 * self.n)])
    elif self.validation:
        self.train_rows, self.validation_rows = np.split(np.random.choice(self.n, self.n, replace=False),
                                                         [int(.6 * self.n)])
        self.test_rows = None
    elif self.test:
        self.train_rows, self.test_rows = np.split(np.random.choice(self.n, self.n, replace=False),
                                                   [int(.6 * self.n)])
        self.validation_rows = None
    else:
        self.train_rows = np.random.choice(self.n, self.n, replace=False)
        self.validation_rows = None
        self.test_rows = None


def fit(self, predictors_cont, predictors_qual, labels, iter=100, **kwargs):
    """
    Fits the Glmdisc object.

    .. todo:: On regarde si des modalités sont présentes dans validation et pas dans train

    :param numpy.ndarray predictors_cont:
        Continuous predictors to be discretized in a numpy
        "numeric" array. Can be provided either here or with
        the __init__ method.
    :param numpy.ndarray predictors_qual:
        Categorical features which levels are to be merged
        (also in a numpy "string" array). Can be provided
        either here or with the __init__ method.
    :param numpy.ndarray labels:
        Boolean (0/1) labels of the observations. Must be of
        the same length as predictors_qual and predictors_cont
        (numpy "numeric" array).
    """
    # iter doit être suffisamment grand
    if iter <= 10:
        msg = 'iter is too low / negative. Please set 10 < iter < 100 000'
        logger.error(msg)
        raise ValueError(msg)

    # iter doit être suffisamment petit
    if iter >= 100000:
        msg = 'iter is too high, it will take years to finish! Please set 10 < iter < 100 000'
        logger.error(msg)
        raise ValueError(msg)

    self.iter = iter

    _check_args(predictors_cont, predictors_qual, labels)

    self.predictors_cont = predictors_cont
    self.predictors_qual = predictors_qual
    self.labels = labels

    # Calcul des variables locales utilisées dans la suite
    continu_complete_case = self._calculate_shape()

    # Initial random "discretization"
    edisc, predictors_trans = self._init_disc(continu_complete_case)

    # Random splitting
    self._split()

    if self.algorithm == "SEM":
        _fit_sem(self, edisc, predictors_trans, continu_complete_case, **kwargs)
    elif self.algorithm == "NN":
        _fit_nn(self, predictors_trans, **kwargs)

    return self
