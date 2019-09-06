#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 16:17:34 2019

@author: adrien
"""


def bestFormula(self):
    """Returns the best formula found by the MCMC."""
    emap_best = self.discretize(self.predictors_cont, self.predictors_qual)

    # Traitement des variables continues

    # Calculate shape of predictors (re-used multiple times)
    try:
        d1 = self.predictors_cont.shape[1]
    except AttributeError:
        d1 = 0

    try:
        d2 = self.predictors_qual.shape[1]
    except AttributeError:
        d2 = 0

    best_disc = []

    for j in range(d1):
        best_disc.append([])
        for k in np.unique(emap_best[:, j]):
            best_disc[j].append(
                    np.nanmin(self.predictors_cont[emap_best[:, j] == k, j]))
        del best_disc[j][np.where(best_disc[j] == np.nanmin(best_disc[j]))[0][0]]
        print("Cut-points found for continuous variable", j + 1, "are", best_disc[j])

    for j in range(d2):
        best_disc.append([])
        for k in np.unique(emap_best[:, j + d1]):
            best_disc[j + d1].append(np.unique(self.predictors_qual[emap_best[:, j + d1] == k, j]))
        print("Regroupments made for categorical variable", j+1, "are", best_disc[j + d1])

    return best_disc
