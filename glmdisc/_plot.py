#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""plot method for class glmdisc
"""
import pandas as pd
import matplotlib.pyplot as plt
from pygam import LogisticGAM
import numpy as np
from loguru import logger


def plot(self,
         predictors_cont_number="all",
         predictors_qual_number="all",
         plot_type="disc"):
    """
    Plots the stepwise function associating the continuous features to their
    discretization, the groupings made and the interactions.

    .. todo:: explain this function better + clean + test args

    :param predictors_cont_number:
        Which continuous variable(s) should be plotted
        (between 1 and the number of columns in
        predictors_cont).
    :type predictors_cont_number: str or int

    :param predictors_qual_number:
        Which categorical variable(s) should be plotted
        (between 1 and the number of columns in
        predictors_qual).
    :type predictors_qual_number: str or int

    :param plot_type: disc or logodd
    :type plot_type: str
    """
    emap = self.discretize(self.predictors_cont,
                           self.predictors_qual).astype(str)

    if plot_type == "disc":

        if predictors_cont_number == "all":
            for j in range(self.d_cont):
                plt.plot(self.predictors_cont[:, j].reshape(-1, 1),
                         emap.astype(str)[:, j], 'ro')
                plt.show(block=False)

        if predictors_qual_number == "all":
            for j in range(self.d_qual):
                plt.plot(self.predictors_qual[:, j].reshape(-1, 1),
                         emap.astype(str)[:, j + self.d_cont], 'ro')
                plt.show(block=False)

        if not predictors_cont_number == "all":
            if type(predictors_cont_number) == int and predictors_cont_number > 0:
                plt.plot(self.predictors_cont[:,
                         predictors_cont_number - 1].reshape(-1, 1),
                         emap.astype(str)[:, predictors_cont_number - 1], 'ro')
                plt.show(block=False)
            else:
                logger.warning("A single int (more than 0 and less than the "
                               "number of columns in predictors_cont) must be "
                               "provided for predictors_cont_number")

        if not predictors_qual_number == "all":
            if type(predictors_qual_number) == int and predictors_qual_number > 0:
                plt.plot(self.predictors_qual[:,
                         predictors_qual_number - 1].reshape(-1, 1),
                         emap.astype(str)[:, predictors_qual_number - 1 + self.d_cont],
                         'ro')
                plt.show(block=False)
            else:
                logger.warning("A single int (more than 0 and less than the "
                               "number of columns in predictors_qual) must be "
                               "provided for predictors_qual_number")

    elif plot_type == "logodd":

        # Gérer les manquants dans le GAM
        lignes_completes = np.invert(
            np.isnan(self.predictors_cont).sum(axis=1).astype(bool))

        # Fit du GAM sur tout le monde
        gam = LogisticGAM(dtype=['numerical' for _ in range(self.d_cont)] + ['categorical' for _ in range(
            self.d_qual)]).fit(
            pd.concat([pd.DataFrame(self.predictors_cont[lignes_completes, :]).apply(
                lambda x: x.astype('float')),
                pd.DataFrame(self.predictors_qual[lignes_completes, :]).apply(
                    lambda x: x.astype('category'))], axis=1), self.labels[lignes_completes])

        # Quelles que soient les valeurs de predictors_cont_number et
        # predictors_qual_number, on plot tout pour l'instant
        plt.figure()
        fig, axs = plt.subplots(1, self.d_cont + self.d_qual)
        plt.rcParams['figure.figsize'] = (28, 8)
        for i, ax in enumerate(axs):
            try:
                XX = gam.generate_X_grid(term=i)
                ax.plot(XX[:, i], gam.partial_dependence(term=i, X=XX))
                ax.plot(XX[:, i], gam.partial_dependence(term=i, X=XX, width=.95)[1], c='r', ls='--')
            except ValueError:  # pragma: no cover
                continue
        plt.show(block=False)
