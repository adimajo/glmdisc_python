#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fit method for the glmdisc class.
"""
import numpy as np
import sklearn as sk
import sklearn.preprocessing
import sklearn.linear_model
from scipy import stats
from collections import Counter
from math import log
from copy import deepcopy
from glmdisc import vectorized_multinouilli

NUMPY_NDARRAY_INPUTS = 'glmdisc only supports numpy.ndarray inputs'


def _check_args(predictors_cont, predictors_qual, labels, check_labels=True):
    """
    Checks inputs

    :param numpy.array predictors_cont: continuous predictors
    :type predictors_cont: numpy.array
    :param numpy.array predictors_qual: categorical predictors
    :type predictors_qual: numpy.array
    :param labels: binary labels
    :type labels: numpy.array
    """
    # Tester la présence de labels
    if predictors_cont is not None and not isinstance(predictors_cont, np.ndarray):
        raise ValueError(NUMPY_NDARRAY_INPUTS)
    if predictors_qual is not None and not isinstance(predictors_qual, np.ndarray):
        raise ValueError(NUMPY_NDARRAY_INPUTS)
    if check_labels and not isinstance(labels, np.ndarray):
        raise ValueError(NUMPY_NDARRAY_INPUTS)

    # Tester la présence d'au moins qual ou cont
    if predictors_cont is None and predictors_qual is None:
        raise ValueError(('You must provide either qualitative or quantitative '
                         'features'))

    # Tester la présence de prédicteurs catégoriels et de même longueur que labels
    if check_labels and ((predictors_cont is not None and predictors_cont.shape[0] != labels.shape[0]) or
                         (predictors_qual is not None and predictors_qual.shape[0] != labels.shape[0])):
        raise ValueError('Predictors and labels must be of same size')


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

    # Gérer les manquants des variables continues, dans un premier temps
    # comme une modalité à part
    if self.predictors_cont is not None:
        continu_complete_case = np.invert(np.isnan(self.predictors_cont))
    else:
        continu_complete_case = None
    return continu_complete_case


def _calculate_criterion(self, emap, model_emap, current_encoder_emap):
    if self.criterion in ['aic', 'bic']:
        loglik = -sk.metrics.log_loss(self.labels[self.train],
                                      model_emap.predict_proba(
                                          X=current_encoder_emap.transform(
                                              emap[self.train, :].astype(str))),
                                      normalize=False)
        if self.validation:
            return loglik

    if self.criterion == 'aic' and not self.validation:
        return -(2 * model_emap.coef_.shape[1] - 2 * loglik)

    if self.criterion == 'bic' and not self.validation:
        return -(log(self.n) * model_emap.coef_.shape[1] - 2 * loglik)

    if self.criterion == 'gini' and self.validation:
        return sk.metrics.roc_auc_score(
            self.labels[self.validate], model_emap.predict_proba(
                X=current_encoder_emap.transform(
                    emap[self.validate, :].astype(str)))[:, 1:])

    if self.criterion == 'gini' and not self.validation:
        return sk.metrics.roc_auc_score(
            self.labels[self.train], model_emap.predict_proba(
                X=current_encoder_emap.transform(
                    emap[self.train, :].astype(str)))[:, 1:])


def _init_disc(self, continu_complete_case):
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
    if self.validation and self.test:
        self.train, self.validate, self.test_rows = np.split(np.random.choice(self.n,
                                                                              self.n,
                                                                              replace=False),
                                                             [int(.6 * self.n), int(.8 * self.n)])
    elif self.validation:
        self.train, self.validate = np.split(np.random.choice(self.n, self.n, replace=False),
                                             [int(.6 * self.n)])
        self.test_rows = None
    elif self.test:
        self.train, self.test_rows = np.split(np.random.choice(self.n, self.n, replace=False),
                                              [int(.6 * self.n)])
        self.validate = None
    else:
        self.train = np.random.choice(self.n, self.n, replace=False)
        self.validate = None
        self.test_rows = None


def fit(self, predictors_cont, predictors_qual, labels):
    """
    Fits the Glmdisc object.

    .. todo:: On regarde si des modalités sont présentes dans validation et pas dans train

    .. todo:: Refactor due to complexity

    .. todo:: Add glmdisc-NN

    :param numpy.array predictors_cont:
        Continuous predictors to be discretized in a numpy
        "numeric" array. Can be provided either here or with
        the __init__ method.
    :param numpy.array predictors_qual:
        Categorical features which levels are to be merged
        (also in a numpy "string" array). Can be provided
        either here or with the __init__ method.
    :param numpy.array labels:
        Boolean (0/1) labels of the observations. Must be of
        the same length as predictors_qual and predictors_cont
        (numpy "numeric" array).
    """
    _check_args(predictors_cont, predictors_qual, labels)

    self.predictors_cont = predictors_cont
    self.predictors_qual = predictors_qual
    self.labels = labels

    # Calcul des variables locales utilisées dans la suite
    continu_complete_case = self._calculate_shape()

    # Initialization for following the performance of the discretization
    current_best = 0

    # Initial random "discretization"
    edisc, predictors_trans = self._init_disc(continu_complete_case)

    emap = np.ndarray.copy(edisc)

    model_edisc = sk.linear_model.LogisticRegression(solver='liblinear',
                                                     C=1e40,
                                                     tol=0.001,
                                                     max_iter=25,
                                                     warm_start=False)

    model_emap = sk.linear_model.LogisticRegression(solver='liblinear',
                                                    C=1e40,
                                                    tol=0.001,
                                                    max_iter=25,
                                                    warm_start=False)

    current_encoder_edisc = sk.preprocessing.OneHotEncoder()

    current_encoder_emap = sk.preprocessing.OneHotEncoder()

    # Initialisation link et m
    link = [None] * (self.d_cont + self.d_qual)
    m = [None] * (self.d_cont + self.d_qual)

    for j in range(self.d_cont):
        link[j] = sk.linear_model.LogisticRegression(C=1e40,
                                                     multi_class='multinomial',
                                                     solver='newton-cg',
                                                     max_iter=25,
                                                     tol=0.001,
                                                     warm_start=False)

    # Random splitting
    self._split()

    # Itérations MCMC
    for i in range(self.iter):

        # Recalcul des matrices disjonctives
        current_encoder_edisc.fit(X=edisc.astype(str))
        current_encoder_emap.fit(X=emap.astype(str))

        # Apprentissage p(y|e) et p(y|emap)
        model_edisc.fit(X=current_encoder_edisc.transform(
            edisc[self.train, :].astype(str)),
            y=self.labels[self.train])

        model_emap.fit(X=current_encoder_emap.transform(
            emap[self.train, :].astype(str)),
            y=self.labels[self.train])

        # Calcul du critère
        self.criterion_iter.append(self._calculate_criterion(emap,
                                                             model_emap,
                                                             current_encoder_emap))

        # Mise à jour éventuelle du meilleur critère
        if self.criterion_iter[i] >= self.criterion_iter[current_best]:
            # Update current best logistic regression
            self.best_reglog = deepcopy(model_emap)
            self.best_link = [deepcopy(link_model) for link_model in link]
            current_best = i
            self.best_encoder_emap = deepcopy(current_encoder_emap)
            self.best_encoder_emap.set_params(handle_unknown='ignore')
            self.performance = self.criterion_iter[current_best]

        for j in range(self.d_cont + self.d_qual):
            m[j] = np.unique(edisc[:, j])

        # On construit la base disjonctive nécessaire au modèle de régression
        # logistique
        base_disjonctive = current_encoder_edisc.transform(
            X=edisc.astype(str)).toarray()

        # On boucle sur les variables pour le tirage de e^j | reste
        for j in np.random.permutation(self.d_cont + self.d_qual):
            # On commence par les quantitatives
            if j < self.d_cont:
                # On apprend e^j | x^j
                link[j].fit(y=edisc[self.train, :][continu_complete_case[self.train, j], j],
                            X=predictors_cont[self.train, :][continu_complete_case[self.train, j], j].reshape(-1, 1))

                y_p = np.zeros((self.n, len(m[j])))

                # On calcule y | e^{-j} , e^j
                for k in range(len(m[j])):
                    modalites = np.zeros((self.n, len(m[j])))
                    modalites[:, k] = np.ones((self.n, ))
                    y_p[:, k] = model_edisc.predict_proba(np.column_stack(
                        (base_disjonctive[:, 0:(sum(list(map(len, m[0:j]))))],
                         modalites,
                         base_disjonctive[:, (sum(list(map(len,
                                                           m[0:(j + 1)])))):(sum(list(map(len, m))))]))
                    )[:, 1] * (2 * np.ravel(self.labels) - 1) - np.ravel(self.labels) + 1

                # On calcule e^j | x^j sur tout le monde
                t = link[j].predict_proba(
                    self.predictors_cont[(continu_complete_case[:, j]), j].reshape(-1, 1))

                # On gère le cas où une ou plusieurs modalités ont disparu de train
                if t.shape[1] < y_p.shape[1]:
                    modalites_manquantes = np.in1d(m[j],
                                                   np.unique(edisc[self.train, :][continu_complete_case[self.train, j],
                                                                                  j]))
                    t2 = np.zeros((sum(continu_complete_case[:, j]), len(m[j])))
                    t2[:, modalites_manquantes] = t
                    t = t2.copy()

                # On met à jour emap^j
                emap[(continu_complete_case[:, j]), j] = np.argmax(t, axis=1)
                emap[np.invert(continu_complete_case[:, j]), j] = m[j][-1]

                # On calcule e^j | reste
                if np.invert(continu_complete_case[:, j]).sum() == 0:
                    t = t * y_p
                else:
                    t = t[:, 0:(len(m[j]) - 1)] * y_p[continu_complete_case[:, j], 0:(len(m[j]) - 1)]

                t = t / (t.sum(axis=1)[:, None])

                # On met à jour e^j
                edisc[continu_complete_case[:, j], j] = vectorized_multinouilli(t, m[j])
                edisc[np.invert(continu_complete_case[:, j]), j] = max(m[j])

            # Variables qualitatives
            else:
                # On fait le tableau de contingence e^j | x^j
                link[j] = Counter([tuple(element) for element in np.column_stack(
                    (predictors_trans[self.train, j - self.d_cont],
                     edisc[self.train, j]))])

                y_p = np.zeros((self.n, len(m[j])))

                # On calcule y | e^{-j} , e^j
                for k in range(len(m[j])):
                    modalites = np.zeros((self.n, len(m[j])))
                    modalites[:, k] = np.ones((self.n, ))

                    y_p[:, k] = model_edisc.predict_proba(np.column_stack(
                        (base_disjonctive[:, 0:(sum(list(map(len, m[0:j]))))],
                         modalites,
                         base_disjonctive[:,
                         (sum(list(map(len, m[0:(j + 1)])))):(sum(list(map(len, m))))]))
                    )[:, 1] * (2 * np.ravel(self.labels) - 1) - np.ravel(self.labels) + 1

                t = np.zeros((self.n, int(len(m[j]))))

                # On calcule e^j | x^j sur tout le monde
                for i2 in range(self.n):
                    for k in range(int(len(m[j]))):
                        t[i2, k] = link[j][(predictors_trans[i2, j - self.d_cont], k)] / self.n

                # On met à jour emap^j
                emap[:, j] = np.argmax(t, axis=1)

                # On calcule e^j | reste
                t = t * y_p
                t = t / (t.sum(axis=1)[:, None])

                edisc[:, j] = vectorized_multinouilli(t, m[j])
