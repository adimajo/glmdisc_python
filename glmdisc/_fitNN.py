#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
import numpy as np
import sklearn as sk
import sklearn.preprocessing
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
sess = tf.Session()


def initialize_neural_net(self, m_quant, m_qual):
    liste_inputs_quant = [None] * self.d_cont
    liste_inputs_qual = [None] * self.d_qual

    liste_layers_quant = [None] * self.d_cont
    liste_layers_qual = [None] * self.d_qual

    liste_layers_quant_inputs = [None] * self.d_cont
    liste_layers_qual_inputs = [None] * self.d_qual

    for i in range(self.d_cont):
        liste_inputs_quant[i] = Input((1, ))
        liste_layers_quant[i] = Dense(m_quant[i],
                                      activation='softmax')
        liste_layers_quant_inputs[i] = liste_layers_quant[i](
            liste_inputs_quant[i])

    for i in range(self.d_qual):
        liste_inputs_qual[i] = Input((len(np.unique(x_qual[:, i])), ))
        if len(np.unique(x_qual[:, i])) > m_qual[i]:
            liste_layers_qual[i] = Dense(
                m_qual[i],
                activation='softmax',
                use_bias=False)
        else:
            liste_layers_qual[i] = Dense(
                len(np.unique(x_qual[:, i])),
                activation='softmax',
                use_bias=False)

        liste_layers_qual_inputs[i] = liste_layers_qual[i](
            liste_inputs_qual[i])

    return ([
        liste_inputs_quant, liste_layers_quant, liste_layers_quant_inputs,
        liste_inputs_qual, liste_layers_qual, liste_layers_qual_inputs
    ])


def from_layers_to_proba_training(d1, d2, liste_layers_quant, liste_layers_qual):

    results = [None] * (d1 + d2)

    for j in range(d1):
        results[j] = tf.function([liste_layers_quant[j].input],
                                 [liste_layers_quant[j].output])(
            [x_quant[:, j, np.newaxis]])

    for j in range(d2):
        results[j + d1] = tf.function([liste_layers_qual[j].input],
                                      [liste_layers_qual[j].output])(
            [liste_qual_arrays[j]])

    return results


def from_weights_to_proba_test(d1, d2, m_quant, m_qual, history, x_quant_test, x_qual_test, n_test):

    results = [None] * (d1 + d2)

    for j in range(d1):
        results[j] = np.zeros((n_test, m_quant[j]))
        for i in range(m_quant[j]):
            results[j][:, i] = history.best_weights[j][1][i] + history.best_weights[j][0][0][i] * x_quant_test[:, j]

    for j in range(d2):
        results[j + d1] = np.zeros((n_test, history.best_weights[j + d1][0].shape[1]))
        for i in range(history.best_weights[j + d1][0].shape[1]):
            for k in range(n_test):
                results[j + d1][k, i] = history.best_weights[j + d1][0][x_qual_test[k, j], i]

    return results


def evaluate_disc(type, d1, d2, misc):
    if type == "train":
        proba = from_layers_to_proba_training(d1, d2, misc[0], misc[1])
    else:
        proba = from_weights_to_proba_test(d1, d2, misc[0], misc[1], misc[2], misc[3], misc[4], misc[5])

    results = [None] * (d1 + d2)

    if type=="train":
        X_transformed = np.ones((n, 1))
    else:
        X_transformed = np.ones((n_test, 1))

    for j in range(d1 + d2):
        if type == "train":
            results[j] = np.argmax(proba[j][0], axis=1)
        else:
            results[j] = np.argmax(proba[j], axis=1)
        X_transformed = np.concatenate(
            (X_transformed, sk.preprocessing.OneHotEncoder(categories='auto', sparse=False, handle_unknown="ignore").fit_transform(
                X=results[j].reshape(-1, 1))),
            axis=1)

    proposed_logistic_regression = sk.linear_model.LogisticRegression(
        fit_intercept=False, solver = "lbfgs", C=1e20, tol=1e-8, max_iter=50)

    if type == "train":
        proposed_logistic_regression.fit(X=X_transformed, y=y.reshape((n, )))
        performance = 2 * sk.metrics.log_loss(
            y,
            proposed_logistic_regression.predict_proba(X=X_transformed)[:, 1],
            normalize=False
        ) + proposed_logistic_regression.coef_.shape[1] * np.log(n)
        predicted = proposed_logistic_regression.predict_proba(X_transformed)[:, 1]

    else:
        proposed_logistic_regression.fit(X=X_transformed, y=y_test.reshape((n_test, )))
        performance = 2 * sk.metrics.roc_auc_score(y_test,
                                                   proposed_logistic_regression.predict_proba(X_transformed)[:, 1]) - 1
        predicted = proposed_logistic_regression.predict_proba(X_transformed)[:, 1]

    return performance, predicted
