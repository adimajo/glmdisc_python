#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""linearity examples for README.
"""
import random
import numpy as np
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
p = 1 / (1 + np.exp(-3 * x ** 5))
y = np.random.binomial(1, p)

# Linear logistic regression fitting
logreg_cont = sk.linear_model.LogisticRegression()
logreg_cont.fit(X=x, y=y.ravel())
logreg_cont_pred = logreg_cont.predict_proba(X=x)[:, 1]
print(np.concatenate([p, logreg_cont_pred.reshape(-1, 1)], axis=1))

# "True" logistic regression fitting
logreg_true = sk.linear_model.LogisticRegression()
logreg_true.fit(X=x ** 5, y=y.ravel())
logreg_true_pred = logreg_true.predict_proba(X=x ** 5)[:, 1]
print(np.concatenate([p, logreg_cont_pred.reshape(-1, 1), logreg_true_pred.reshape(-1, 1)], axis=1))

# Discretized logistic regression fitting
logreg_disc = glmdisc.Glmdisc(iter=100)
logreg_disc.fit(predictors_cont=x, predictors_qual=None, labels=y.ravel())
logreg_disc_pred = logreg_disc.predict(predictors_cont=x, predictors_qual=None)[:, 1]
print(np.concatenate([p, logreg_cont_pred.reshape(-1, 1), logreg_true_pred.reshape(-1, 1), logreg_disc_pred.reshape(-1, 1)], axis=1))

# Plots
fig, ax = plt.subplots()
ax.plot(x, p, 'bo', label="True probability")
ax.plot(x, logreg_cont_pred.reshape(-1, 1), 'ro', label="Continuous model probability")
ax.plot(x, logreg_disc_pred.reshape(-1, 1), 'go', label="Discretized model probability")
legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')
plt.show(block=False)
