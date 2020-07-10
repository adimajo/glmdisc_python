# -*- coding: utf-8 -*-
"""This module is dedicated to preprocessing tasks for logistic regression and
post-learning graphical tools.

.. autosummary::
    :toctree:

    Glmdisc
    Glmdisc.check_is_fitted
    Glmdisc.best_formula
    Glmdisc.discrete_data
    Glmdisc.discretize
    Glmdisc.discretize_dummy
    Glmdisc.fit
    Glmdisc.plot
    Glmdisc.predict
    Glmdisc.generate_data
    NotFittedError
"""
import numpy as np
import sklearn as sk
from loguru import logger


class NotFittedError(sk.exceptions.NotFittedError):
    """Exception class to raise if estimator is used before fitting.
    This class inherits from both NotFittedError from sklearn which
    itself inherits from ValueError and AttributeError to help with
    exception handling and backward compatibility.
    """


def vectorized_multinouilli(prob_matrix, items):
    """
    A vectorized version of multinouilli sampling.

    .. todo:: check that the number of columns of prob_matrix is the same as the number of elements in items

    :param prob_matrix: A probability matrix of size n (number of training
        examples) * m[j] (the factor levels to sample from).
    :type prob_matrix: numpy.array

    :param list items: The factor levels to sample from.

    :returns: The drawn factor levels for each observation.
    :rtype: numpy.array
    """

    s = prob_matrix.cumsum(axis=1)
    r = np.random.rand(prob_matrix.shape[0]).reshape((-1, 1))
    k = (s < r).sum(axis=1)
    return items[k]


class Glmdisc:
    """
    This class implements a supervised multivariate discretization method,
    factor levels grouping and interaction discovery for logistic regression.

    .. attribute:: test

        Boolean (T/F) specifying if a test set is required.
        If True, the provided data is split to provide 20%
        of observations in a test set and the reported
        performance is the Gini index on test set.

       :type: bool

    .. attribute:: validation

        Boolean (T/F) specifying if a validation set is
        required. If True, the provided data is split to
        provide 20% of observations in a validation set
        and the reported performance is the Gini index on
        the validation set (if no test=False). The quality
        of the discretization at each step is evaluated
        using the Gini index on the validation set, so
        criterion must be set to "gini".

        :type: bool

    .. attribute:: criterion

        The criterion to be used to assess the
        goodness-of-fit of the discretization: "bic" or
        "aic" if no validation set, else "gini".

        :type: str

    .. attribute:: iter

        Number of MCMC steps to perform. The more the
        better, but it may be more intelligent to use
        several MCMCs. Computation time can increase
        dramatically.

        :type: int

    .. attribute:: m_start

        Number of initial discretization intervals for all
        variables. If :code:`m_start` is bigger than the number of
        factor levels for a given variable in
        predictors_qual, m_start is set (for this variable
        only) to this variable's number of factor levels.

        :type: int

    .. attribute:: criterion_iter

        The value of the criterion wished to be optimized
        over the iterations.

        :type: list

    .. attribute:: best_link

        The best link function between the original
        features and their quantized counterparts that
        allows to quantize the data after learning.

        :type: list

    .. attribute:: best_reglog:

        The best logistic regression on quantized data found with best_link.

        :type: sklearn.linear_model.LogisticRegression

    .. attribute:: affectations

        The label encoder of each original feature.
        best_encoder_emap (list): The label encoder of each of the best_link.

        :type: list

    .. attribute:: performance:

        The best 'criterion' obtained.

        :type: list

    .. attribute:: splitting

        The line rows corresponding to the splits.

        :type: list
    """

    def __init__(self, test=True, validation=True, criterion="bic", iter=100, m_start=20):
        """
        Initializes self by checking if its arguments are appropriately specified.

        :param bool test: Boolean specifying if a test set is required.
                            If True, the provided data is split to provide 20%
                            of observations in a test set and the reported
                            performance is the Gini index on test set.

        :param bool validation: Boolean (T/F) specifying if a validation set is
                            required. If True, the provided data is split to
                            provide 20% of observations in a validation set
                            and the reported performance is the Gini index on
                            the validation set (if no test=False). The quality
                            of the discretization at each step is evaluated
                            using the Gini index on the validation set, so
                            criterion must be set to "gini".

        :param str criterion: The criterion to be used to assess the
                            goodness-of-fit of the discretization: "bic" or
                            "aic" if no validation set, else "gini".

        :param int iter: Number of MCMC steps to perform. The more the
                            better, but it may be more intelligent to use
                            several MCMCs. Computation time can increase
                            dramatically. Defaults to 100.

        :param int m_start: Number of initial discretization intervals for all
                            variables. If :code:`m_start` is bigger than the number of
                            factor levels for a given variable in
                            :code:`predictors_qual`, :code:`m_start` is set (for this variable
                            only) to this variable's number of factor levels. Defaults to 20.

        .. todo:: Gérer un try catch pour warm start ?
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

        if not validation and criterion == 'gini':
            logger.warning('Using Gini index on training set might yield an overfitted model')

        if validation and criterion in ['aic', 'bic']:
            logger.warning('No need to penalize the log-likelihood when a validation set is used. '
                           'Using log-likelihood instead.')

        # Attributes from parameters from __init__
        self.test = test
        self.validation = validation
        self.criterion = criterion
        self.iter = iter
        self.m_start = m_start

        # Attributes from fit
        self.n = 0
        self.d_cont = 0
        self.d_qual = 0

        self.predictors_cont = None
        self.predictors_qual = None
        self.predictors_labels = None

        self.criterion_iter = []
        self.best_link = []
        self.best_reglog = None
        self.affectations = []
        self.best_encoder_emap = None
        self.performance = -np.inf
        self.train = np.array([])
        self.validate = np.array([])
        self.test_rows = np.array([])

    def check_is_fitted(self):
        """Perform is_fitted validation for estimator.
        Checks if the estimator is fitted by verifying the presence of
        fitted attributes (ending with a trailing underscore) and otherwise
        raises a NotFittedError with the given message.
        This utility is meant to be used internally by estimators themselves,
        typically in their own predict / transform methods.
        """
        try:
            sk.utils.validation.check_is_fitted(self.best_reglog)
            for link in self.best_link:
                if isinstance(link, sk.linear_model.LogisticRegression):
                    sk.utils.validation.check_is_fitted(link)
        except sk.exceptions.NotFittedError as e:
            raise NotFittedError(str(e) + " If you did call fit, try increasing iter: "
                                          "it means it did not find a better solution than the random initialization.")

    # Imported methods
    from ._bestFormula import best_formula
    from ._discreteData import discrete_data
    from ._discretize import discretize
    from ._discretizeDummy import discretize_dummy
    from ._fit import fit, _calculate_shape, _calculate_criterion, _init_disc, _split
    from ._plot import plot
    from ._predict import predict
    from ._generateData import generate_data
