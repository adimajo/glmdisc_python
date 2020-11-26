#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""best_formula method for glmdisc class.
"""
from heapq import nlargest
from itertools import combinations

import numpy as np
from loguru import logger


def _best_formula_cont_sem(self, emap_best):

    best_disc = []

    for j in range(self.d_cont):
        best_disc.append([])
        for k in np.unique(emap_best[:, j]):
            best_disc[j].append(
                np.nanmin(self.predictors_cont[emap_best[:, j] == k, j]))
        del best_disc[j][np.where(best_disc[j] == np.nanmin(best_disc[j]))[0][0]]
        if len(best_disc[j]) < 1:
            logger.info("No cut-points found for continuous variable " + str(j) + ".")
        else:
            logger.info("Cut-points found for continuous variable " + str(j) + " are "
                        + str(best_disc[j]))

    return best_disc


def _best_formula_cont_nn(self, emap_best):
    best_disc = []
    best_weights = self.callbacks[1].best_weights
    for j in range(self.d_cont):
        list_modalites = np.unique(emap_best[:, j])
        if len(list_modalites) <= 1:
            best_disc.append([])
            logger.info("No cut-points found for continuous variable " + str(j) + ".")
            continue
        points_coupure = []
        proba_points_coupure = []
        for h1, h2 in combinations(list_modalites, r=2):
            point_coupure = (best_weights[j][1][h1] - best_weights[j][1][h2]) / (best_weights[j][0][0][h2] -
                                                                                 best_weights[j][0][0][h1])
            points_coupure.append(point_coupure)
            proba_point_coupure_num = np.exp(best_weights[j][1][h1] + best_weights[j][0][0][h1] * point_coupure)
            proba_points_coupure_denom = np.sum([np.exp(best_weights[j][1][h] +
                                                        best_weights[j][0][0][h] * point_coupure)
                                                 for h in list_modalites])
            proba_points_coupure.append(proba_point_coupure_num / proba_points_coupure_denom)

        largest = nlargest(len(list_modalites) - 1, proba_points_coupure)
        best_disc.append(np.array(points_coupure)[proba_points_coupure >= np.nanmin(largest)])

        logger.info("Cut-points found for continuous variable " + str(j) + " are "
                    + str(best_disc[j]))
    return best_disc


def best_formula(self):
    """
    Returns the best quantization found by the MCMC and prints it.

    :returns:
        A list of cutpoints (continuous features) or groupings
        (categorical features).

    :rtype: list
    """
    emap_best = self.discretize(self.predictors_cont, self.predictors_qual)

    if self.d_cont > 0:
        if self.algorithm == "SEM":
            best_disc = _best_formula_cont_sem(self, emap_best)

        elif self.algorithm == "NN":
            best_disc = _best_formula_cont_nn(self, emap_best)
    else:
        best_disc = []

    for j in range(self.d_qual):
        best_disc.append([])
        les_modalites = np.unique(emap_best[:, j + self.d_cont])
        if len(les_modalites) <= 1:
            logger.info("No regroupments made for categorical variable " + str(j) + ".")
            continue
        for k in les_modalites:
            best_disc[j + self.d_cont].append(np.unique(self.predictors_qual[emap_best[:, j + self.d_cont] == k, j]))
        logger.info("Regroupments made for categorical variable " + str(j) + " are "
                    + str(best_disc[j + self.d_cont]))

    return best_disc
