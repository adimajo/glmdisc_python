#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fit function for SEM algorithm
"""
from collections import Counter
from copy import deepcopy
from math import log

import numpy as np
import sklearn as sk
import sklearn.linear_model
from loguru import logger

from glmdisc import _vectorized_multinouilli


def _calculate_criterion(self, emap, model_emap, current_encoder_emap):
    """
    Calculate current value of optimised criterion

    Parameters
    ----------
    emap: array of current discretization / grouping of size d_cont + d_qual
    model_emap: current logistic regression
    current_encoder_emap: one hot encoder of emap

    Returns
    -------
    criterion value
    """
    if self.criterion in ['aic', 'bic']:
        loglik = -sk.metrics.log_loss(self.labels[self.train_rows],
                                      model_emap.predict_proba(
                                          X=current_encoder_emap.transform(
                                              emap[self.train_rows, :].astype(str))),
                                      normalize=False)
        if self.validation:
            performance = loglik
            logger.info("Current likelihood on validation set: " + str(performance))

    if self.criterion == 'aic' and not self.validation:
        performance = -(2 * model_emap.coef_.shape[1] - 2 * loglik)
        logger.info("Current AIC on train set: " + str(- performance))

    if self.criterion == 'bic' and not self.validation:
        performance = -(log(self.n) * model_emap.coef_.shape[1] - 2 * loglik)
        logger.info("Current BIC on train set: " + str(- performance))

    if self.criterion == 'gini' and self.validation:
        performance = sk.metrics.roc_auc_score(
            self.labels[self.validation_rows], model_emap.predict_proba(
                X=current_encoder_emap.transform(
                    emap[self.validation_rows, :].astype(str)))[:, 1:])
        logger.info("Current Gini on validation set: " + str(performance))

    if self.criterion == 'gini' and not self.validation:
        performance = sk.metrics.roc_auc_score(
            self.labels[self.train_rows], model_emap.predict_proba(
                X=current_encoder_emap.transform(
                    emap[self.train_rows, :].astype(str)))[:, 1:])
        logger.info("Current Gini on training set: " + str(performance))

    if performance is None:
        msg = "Bug encountered, please open an issue at https://github.com/adimajo/glmdisc_python."
        logger.error(msg)
        raise ValueError(msg)

    return performance


def _init_fit_sem(self, edisc):
    # Initialization for following the performance of the discretization
    current_best = 0

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

    # Initializing links and m (list of number of levels)
    link = [None] * (self.d_cont + self.d_qual)
    m = [None] * (self.d_cont + self.d_qual)

    for j in range(self.d_cont):
        link[j] = sk.linear_model.LogisticRegression(C=1e40,
                                                     multi_class='multinomial',
                                                     solver='newton-cg',
                                                     max_iter=25,
                                                     tol=0.001,
                                                     warm_start=False)

    return current_best, emap, model_edisc, model_emap, current_encoder_edisc, current_encoder_emap, link, m


def _fit_sem(self, edisc, predictors_trans, continu_complete_case, **kwargs):
    """
    fit function for SEM algorithm

    :param self: Glmdisc class instance
    :param numpy.ndarray edisc: initial random assignment to factor levels
    :param numpy.ndarray predictors_trans: transformation of categorical features to integers
    """
    if kwargs != {}:
        logger.warning("**kwargs not used for algorithm = 'SEM'")

    current_best, emap, model_edisc, model_emap, current_encoder_edisc, current_encoder_emap, link, m = _init_fit_sem(
        self,
        edisc)

    # MCMC iterations
    for i in range(self.iter):
        logger.info("Iteration " + str(i) + " of SEM algorithm.")
        # Disjonctive matrices
        current_encoder_edisc.fit(X=edisc.astype(str))
        current_encoder_emap.fit(X=emap.astype(str))

        # Learning p(y|q) et p(y|q(x))
        model_edisc.fit(X=current_encoder_edisc.transform(
            edisc[self.train_rows, :].astype(str)),
            y=self.labels[self.train_rows])

        model_emap.fit(X=current_encoder_emap.transform(
            emap[self.train_rows, :].astype(str)),
            y=self.labels[self.train_rows])

        # Criterion calculation
        self.criterion_iter.append(_calculate_criterion(self,
                                                        emap,
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

        # On boucle sur les variables pour le tirage de q_j | reste
        for j in np.random.permutation(self.d_cont + self.d_qual):
            # On commence par les quantitatives
            if j < self.d_cont:
                # On apprend q_j | x_j
                link[j].fit(y=edisc[self.train_rows, :][continu_complete_case[self.train_rows, j], j],
                            X=self.predictors_cont[self.train_rows, :][continu_complete_case[self.train_rows,
                                                                                             j], j].reshape(-1, 1))

                y_p = np.zeros((self.n, len(m[j])))

                # On calcule y | q_{-j} , q_j
                for k in range(len(m[j])):
                    modalites = np.zeros((self.n, len(m[j])))
                    modalites[:, k] = np.ones((self.n, ))
                    y_p[:, k] = model_edisc.predict_proba(np.column_stack(
                        (base_disjonctive[:, 0:(sum(list(map(len, m[0:j]))))],
                         modalites,
                         base_disjonctive[:, (sum(list(map(len,
                                                           m[0:(j + 1)])))):(sum(list(map(len, m))))]))
                    )[:, 1] * (2 * np.ravel(self.labels) - 1) - np.ravel(self.labels) + 1

                # On calcule q_j | x_j sur tout le monde
                t = link[j].predict_proba(
                    self.predictors_cont[(continu_complete_case[:, j]), j].reshape(-1, 1))

                # On gère le cas où une ou plusieurs modalités ont disparu de train
                if t.shape[1] < y_p.shape[1]:
                    modalites_manquantes = np.in1d(m[j],
                                                   np.unique(edisc[self.train_rows,
                                                             :][continu_complete_case[self.train_rows, j], j]))
                    t2 = np.zeros((sum(continu_complete_case[:, j]), len(m[j])))
                    t2[:, modalites_manquantes] = t
                    t = t2.copy()

                # On met à jour qmap^j
                emap[(continu_complete_case[:, j]), j] = np.argmax(t, axis=1)
                emap[np.invert(continu_complete_case[:, j]), j] = m[j][-1]

                # On calcule q_j | reste
                if np.invert(continu_complete_case[:, j]).sum() == 0:
                    t = t * y_p
                else:
                    t = t[:, 0:(len(m[j]) - 1)] * y_p[continu_complete_case[:, j], 0:(len(m[j]) - 1)]

                t = t / (t.sum(axis=1)[:, None])

                # On met à jour q_j
                edisc[continu_complete_case[:, j], j] = _vectorized_multinouilli(t, m[j])
                edisc[np.invert(continu_complete_case[:, j]), j] = max(m[j])

            # Variables qualitatives
            else:
                # On fait le tableau de contingence q_j | x_j
                link[j] = Counter([tuple(element) for element in np.column_stack(
                    (predictors_trans[self.train_rows, j - self.d_cont],
                     edisc[self.train_rows, j]))])

                y_p = np.zeros((self.n, len(m[j])))

                # On calcule y | q_{-j} , q_j
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

                # On calcule q_j | x_j sur tout le monde
                for i2 in range(self.n):
                    for k in range(int(len(m[j]))):
                        t[i2, k] = link[j][(predictors_trans[i2, j - self.d_cont], k)] / self.n

                # On met à jour qmap_j
                emap[:, j] = np.argmax(t, axis=1)

                # On calcule q_j | reste
                t = t * y_p
                t = t / (t.sum(axis=1)[:, None])

                edisc[:, j] = _vectorized_multinouilli(t, m[j])
