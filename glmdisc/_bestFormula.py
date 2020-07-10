#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""best_formula method for glmdisc class.
"""
import numpy as np
from loguru import logger


def best_formula(self):
    """
    Returns the best quantization found by the MCMC and prints it.

    :returns:
        A list of cutpoints (continuous features) or groupings
        (categorical features).

    :rtype: list
    """
    emap_best = self.discretize(self.predictors_cont, self.predictors_qual)

    best_disc = []

    for j in range(self.d_cont):
        best_disc.append([])
        for k in np.unique(emap_best[:, j]):
            best_disc[j].append(
                np.nanmin(self.predictors_cont[emap_best[:, j] == k, j]))
        del best_disc[j][np.where(best_disc[j] == np.nanmin(best_disc[j]))[0][0]]
        logger.info("Cut-points found for continuous variable", j + 1, "are", best_disc[j])

    for j in range(self.d_qual):
        best_disc.append([])
        for k in np.unique(emap_best[:, j + self.d_cont]):
            best_disc[j + self.d_cont].append(np.unique(self.predictors_qual[emap_best[:, j + self.d_cont] == k, j]))
        logger.info("Regroupments made for categorical variable", j + 1, "are", best_disc[j + self.d_cont])

    return best_disc
