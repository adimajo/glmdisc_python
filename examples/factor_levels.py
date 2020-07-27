#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""linearity examples for README.
"""
import random
import numpy as np
import pandas as pd
import sklearn as sk
import sklearn.linear_model
import matplotlib.pyplot as plt
import glmdisc

# Data Generation
seed = 1
random.seed(seed)
n = 10000
d = 1
x = np.array(np.random.uniform(size=(n, d)))
cuts_not_significant = np.arange(start=0.0, stop=1.05, step=0.05)
cuts_significant = np.arange(start=0.0, stop=1.2, step=0.2)
xd_not_significant = pd.cut(x.ravel(), bins=cuts_not_significant, labels=range(0, len(cuts_not_significant) - 1))
xd_significant = pd.cut(x.ravel(), bins=cuts_significant, labels=range(0, len(cuts_significant) - 1))
theta = np.array([np.random.normal(0, 1) for _ in range(len(cuts_significant) - 1)])
log_odd = np.array([0] * n)
for i in range(n):
    log_odd[i] = theta[int(xd_significant[i])]

p = 1 / (1 + np.exp(log_odd))
y = np.random.binomial(1, p)
xd_not_significant_dummy = sk.preprocessing.OneHotEncoder().fit_transform(np.reshape(xd_not_significant, (-1, 1)))
xd_significant_dummy = sk.preprocessing.OneHotEncoder().fit_transform(np.reshape(xd_significant, (-1, 1)))

# Coefficient estimation with cross validation
K = 10
taille = len(xd_not_significant)/K
list_coef = []
for k in range(10):
    lr = sk.linear_model.LogisticRegression(fit_intercept=False)
    xd_train = xd_not_significant_dummy[int(taille*k):int(taille*(k+1)), :]
    y_train = y[int(taille*k):int(taille*(k+1))]
    lr.fit(X=xd_train, y=y_train)
    list_coef.append(lr.coef_)

list_mean = []
list_sd = []
for coef in range(len(cuts_not_significant) - 1):
    list_mean.append(np.mean([x[0][coef] for x in list_coef]))
    list_sd.append(np.std([x[0][coef] for x in list_coef]))

ci = np.multiply(1.96 / np.sqrt(10), list_sd)

# Plot (non-)significance
plt.errorbar(x=list(xd_not_significant.categories),
             y=list_mean,
             yerr=ci,
             color="black",
             capsize=3,
             linestyle="None",
             marker="s",
             markersize=7,
             mfc="black",
             mec="black")

# Grouping
logreg_disc = glmdisc.Glmdisc(iter=100)
logreg_disc.fit(predictors_cont=None,
                predictors_qual=xd_not_significant.to_numpy().reshape((-1, 1)),
                labels=y)

list_coef_glmdisc = []
for k in range(10):
    xd_train = xd_not_significant.to_numpy().reshape((-1, 1))[int(taille*k):int(taille*(k+1)), :]
    y_train = y[int(taille*k):int(taille*(k+1))]
    xd_train_glmdisc = logreg_disc.discretize_dummy(predictors_cont=None, predictors_qual=xd_train)
    lr = sk.linear_model.LogisticRegression(fit_intercept=False)
    lr.fit(X=xd_train_glmdisc, y=y_train)
    list_coef_glmdisc.append(lr.coef_)

list_mean_glmdisc = []
list_sd_glmdisc = []
for coef in range(len(cuts_not_significant) - 1):  # Adapter range
    list_mean_glmdisc.append(np.mean([x[0][coef] for x in list_coef]))
    list_sd_glmdisc.append(np.std([x[0][coef] for x in list_coef]))

ci_glmdisc = np.multiply(1.96 / np.sqrt(10), list_sd_glmdisc)

# Plot significance
