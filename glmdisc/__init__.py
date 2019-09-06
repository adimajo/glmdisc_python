# -*- coding: utf-8 -*-
"""
    This module is dedicated to preprocessing tasks for logistic regression and post-learning graphical tools.
"""

import numpy as np
import pandas as pd
import sklearn as sk
import sklearn.preprocessing
import sklearn.linear_model
import warnings
import matplotlib.pyplot as plt

from scipy import stats
from collections import Counter
from math import log
from pygam import LogisticGAM


def vectorized_multinouilli(prob_matrix, items):
    """A vectorized version of multinouilli sampling.
    """
    s = prob_matrix.cumsum(axis=1)
    r = np.random.rand(prob_matrix.shape[0]).reshape((-1, 1))
    k = (s < r).sum(axis=1)
    return items[k]


class glmdisc:

    """
    This class implements a supervised multivariate discretization method,
    factor levels grouping and interaction discovery for logistic regression.
    """

    def __init__(self, test=True, validation=True, criterion="bic", iter=100, m_start=20):

        """
        Initializes self by checking if its arguments are appropriately specified.

        Keyword arguments:
        test            -- Boolean (T/F) specifying if a test set is required.
                            If True, the provided data is split to provide 20%
                            of observations in a test set and the reported
                            performance is the Gini index on test set.
        validation      -- Boolean (T/F) specifying if a validation set is
                            required. If True, the provided data is split to
                            provide 20% of observations in a validation set
                            and the reported performance is the Gini index on
                            the validation set (if no test=False). The quality
                            of the discretization at each step is evaluated
                            using the Gini index on the validation set, so
                            criterion must be set to "gini".
        criterion       -- The criterion to be used to assess the
                            goodness-of-fit of the discretization: "bic" or
                            "aic" if no validation set, else "gini".
        iter            -- Number of MCMC steps to perform. The more the
                            better, but it may be more intelligent to use
                            several MCMCs. Computation time can increase
                            dramatically.
        m_start         -- Number of initial discretization intervals for all
                            variables. If m_start is bigger than the number of
                            factor levels for a given variable in
                            predictors_qual, m_start is set (for this variable
                            only) to this variable's number of factor levels.
        """

        # Tests des variables d'entrée

        # Le critère doit être un des trois de la liste
        if criterion not in ['gini', 'aic', 'bic']:
            raise ValueError('Criterion must be one of Gini, Aic, Bic')

        # test est bool
        if not type(test) is bool:
            raise ValueError('test must be boolean')

        # validation est bool
        if not type(validation) is bool:
            raise ValueError('validation must be boolean')

        # iter doit être suffisamment grand
        if iter <= 10:
            raise ValueError('iter is too low / negative. Please set 10 < iter < 100 000')

        # iter doit être suffisamment petit
        if iter >= 100000:
            raise ValueError('iter is too high, it will take years to finish! Please set 10 < iter < 100 000')

        # m_start doit être pas déconnant
        if not 2 <= m_start <= 50:
            raise ValueError('Please set 2 <= m_start <= 50')

        if not(validation) and criterion == 'gini':
            warnings.warn('Using Gini index on training set might yield an overfitted model')

        if validation and criterion in ['aic', 'bic']:
            warnings.warn('No need to penalize the log-likelihood when a validation set is used. Using log-likelihood instead.')

        self.test = test
        self.validation = validation
        self.criterion = criterion
        self.iter = iter
        self.m_start = m_start

        self.criterion_iter = []
        self.best_link = []
        self.best_reglog = 0
        self.affectations = []
        self.best_encoder_emap = []

    # Imported methods
    from ._bestFormula import bestFormula
    from ._contData import contData
    from ._discreteData import discreteData
    from ._discretize import discretize
    from ._discretizeDummy import discretizeDummy
    from ._fit import fit
    from ._performance import performance
    from ._plot import plot
    from ._predict import predict

    # Faire un try catch pour warm start ?
