#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fit function for NN algorithm
"""
import numpy as np
from scipy import stats
import sklearn as sk
import sklearn.preprocessing
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import concatenate
from tensorflow.keras import Model
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau
import tensorflow.keras.optimizers
from itertools import chain
from loguru import logger
import matplotlib.pyplot as plt
import pandas as pd


class LossHistory(Callback):
    def __init__(self, d_cont, d_qual, neural_net):
        super().__init__()
        self.d_cont = d_cont
        self.d_qual = d_qual
        self.neural_net = neural_net
        self.best_weights = None
        self.losses = None
        self.best_criterion = None
        self.best_outputs = None

    def on_train_begin(self, logs={}):
        self.losses = []
        self.best_criterion = float("inf")
        self.best_outputs = []

    def on_epoch_end(self, batch, logs={}, plot=True):
        self.losses.append(evaluate_disc("train", self.d_cont, self.d_qual, self.neural_net)[0])
        if len(self.losses) > 5:  # and self.losses[-1] < self.best_criterion:
            self.best_weights = []
            self.best_outputs = []
            self.best_criterion = self.losses[-1]
            for j in range(self.d_cont):
                self.best_weights.append(self.neural_net["liste_layers_quant"][j].get_weights())
                self.best_outputs.append(
                    tf.keras.backend.function([self.neural_net["liste_layers_quant"][j].input],
                                              [self.neural_net["liste_layers_quant"][j].output])(
                        [self.neural_net["predictors_cont"][:, j, np.newaxis]]))
            for j in range(self.d_qual):
                self.best_weights.append(self.neural_net["liste_layers_qual"][j].get_weights())
                self.best_outputs.append(
                    tf.keras.backend.function([self.neural_net["liste_layers_qual"][j].input],
                                              [self.neural_net["liste_layers_qual"][j].output])(
                        [self.neural_net["predictors_qual_dummy"][j]]))

            if plot:
                plt.xlim((0, 1))
                plt.ylim((0, 1))
                print(self.best_outputs)
                print(self.best_outputs[0])
                print(self.best_outputs[0][0])
                print(self.best_outputs[0][0].shape)
                for k in range(self.best_outputs[0][0].shape[1]):
                    plt.plot(np.sort(self.neural_net["predictors_cont"][:, 0]), self.best_outputs[0][0][np.argsort(self.neural_net["predictors_cont"][:, 0]), k],
                             color=['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'][k + 2])
                plt.show()


def initialize_neural_net(self, predictors_qual_dummy):
    """

    """
    #

    m_cont = [self.m_start] * self.d_cont
    m_qual = [self.m_start] * (self.d_cont + self.d_qual)

    for j in range(self.d_qual):
        num_levels = stats.describe(sk.preprocessing.LabelEncoder().fit(self.predictors_qual[:, j]).transform(
            self.predictors_qual[:, j])).minmax[1]
        if self.m_start > num_levels + 1:
            m_qual[j] = num_levels

    liste_inputs_quant = [None] * self.d_cont
    liste_inputs_qual = [None] * self.d_qual

    liste_layers_quant = [None] * self.d_cont
    liste_layers_qual = [None] * self.d_qual

    liste_layers_quant_inputs = [None] * self.d_cont
    liste_layers_qual_inputs = [None] * self.d_qual

    for i in range(self.d_cont):
        liste_inputs_quant[i] = Input((1, ))
        liste_layers_quant[i] = Dense(m_cont[i],
                                      activation='softmax')
        liste_layers_quant_inputs[i] = liste_layers_quant[i](
            liste_inputs_quant[i])

    for i in range(self.d_qual):
        liste_inputs_qual[i] = Input((len(np.unique(self.predictors_qual[:, i])), ))
        if len(np.unique(self.predictors_qual[:, i])) > m_qual[i]:
            liste_layers_qual[i] = Dense(
                m_qual[i],
                activation='softmax',
                use_bias=False)
        else:
            liste_layers_qual[i] = Dense(
                len(np.unique(self.predictors_qual[:, i])),
                activation='softmax',
                use_bias=False)

        liste_layers_qual_inputs[i] = liste_layers_qual[i](
            liste_inputs_qual[i])

    self.neural_net = {
        "n": self.n,
        "labels": self.labels,
        "predictors_cont": self.predictors_cont,
        "predictors_qual_dummy": predictors_qual_dummy,
        "train": self.train,
        "validate": self.validate,
        "validation": self.validation,
        "criterion": self.criterion,
        "liste_inputs_quant": liste_inputs_quant,
        "liste_layers_quant": liste_layers_quant,
        "liste_layers_quant_inputs": liste_layers_quant_inputs,
        "liste_inputs_qual": liste_inputs_qual,
        "liste_layers_qual": liste_layers_qual,
        "liste_layers_qual_inputs": liste_layers_qual_inputs
    }


def from_layers_to_proba_training(d_cont, d_qual, neural_net):

    results = [None] * (d_cont + d_qual)

    for j in range(d_cont):
        results[j] = tf.keras.backend.function([neural_net["liste_layers_quant"][j].input],
                                               [neural_net["liste_layers_quant"][j].output])(
            [neural_net["predictors_cont"][neural_net["train"], j, np.newaxis]])

    for j in range(d_qual):
        results[j + d_cont] = tf.keras.backend.function([neural_net["liste_layers_qual"][j].input],
                                                        [neural_net["liste_layers_qual"][j].output])(
            [neural_net["predictors_qual_dummy"][j]])

    return results


def from_weights_to_proba_test(d1, d2, m_cont, history, x_quant_test, x_qual_test, n_test):

    results = [None] * (d1 + d2)

    for j in range(d1):
        results[j] = np.zeros((n_test, m_cont[j]))
        for i in range(m_cont[j]):
            results[j][:, i] = history.best_weights[j][1][i] + history.best_weights[j][0][0][i] * x_quant_test[:, j]

    for j in range(d2):
        results[j + d1] = np.zeros((n_test, history.best_weights[j + d1][0].shape[1]))
        for i in range(history.best_weights[j + d1][0].shape[1]):
            for k in range(n_test):
                results[j + d1][k, i] = history.best_weights[j + d1][0][x_qual_test[k, j], i]

    return results


def evaluate_disc(type, d_cont, d_qual, neural_net):
    if type == "train":
        proba = from_layers_to_proba_training(d_cont, d_qual, neural_net)
    else:
        raise NotImplementedError
        # proba = from_weights_to_proba_test(d1, d2, misc[0], misc[1], misc[2], misc[3], misc[4], misc[5])

    results = [None] * (d_cont + d_qual)

    if type == "train":
        X_transformed = np.ones((neural_net["n"], 1))
    else:
        raise NotImplementedError
        # X_transformed = np.ones((n_test, 1))

    for j in range(d_cont + d_qual):
        if type == "train":
            results[j] = np.argmax(proba[j][0], axis=1)
        else:
            raise NotImplementedError
            # results[j] = np.argmax(proba[j], axis=1)
        X_transformed = np.concatenate(
            (X_transformed,
             sk.preprocessing.OneHotEncoder(categories='auto', sparse=False, handle_unknown="ignore").fit_transform(
                 X=results[j].reshape(-1, 1))),
            axis=1)

    proposed_logistic_regression = sk.linear_model.LogisticRegression(
        fit_intercept=False, solver="lbfgs", C=1e20, tol=1e-8, max_iter=50)

    if type == "train":
        proposed_logistic_regression.fit(X=X_transformed, y=neural_net["labels"][neural_net["train"]].reshape(
            (neural_net["train"].shape[0],)))
        if neural_net["criterion"] in ['aic', 'bic']:
            loglik = sk.metrics.log_loss(
                neural_net["labels"],
                proposed_logistic_regression.predict_proba(X=X_transformed)[:, 1],
                normalize=False
            )
            if neural_net["validation"]:
                performance = 2 * loglik
            elif neural_net["criterion"] == "bic":
                performance = 2 * loglik + proposed_logistic_regression.coef_.shape[1] * np.log(neural_net["n"])
            else:
                performance = 2 * loglik + 2 * proposed_logistic_regression.coef_.shape[1]
        else:
            raise NotImplementedError
        predicted = proposed_logistic_regression.predict_proba(X_transformed)[:, 1]

    else:
        raise NotImplementedError
        # proposed_logistic_regression.fit(X=X_transformed, y=y_test.reshape((n_test,)))
        # performance = 2 * sk.metrics.roc_auc_score(y_test,
        #                                            proposed_logistic_regression.predict_proba(
        #                                               X_transformed)[:, 1]) - 1
        # predicted = proposed_logistic_regression.predict_proba(X_transformed)[:, 1]

    print('\n')
    print(performance)
    return performance, predicted


def _fitNN(self, predictors_trans):

    if self.predictors_qual is not None:
        self.one_hot_encoders_nn = []
        predictors_qual_dummy = []
        for j in range(self.d_qual):
            one_hot_encoder = sk.preprocessing.OneHotEncoder()
            predictors_qual_dummy.append(np.squeeze(np.asarray(
                one_hot_encoder.fit_transform(predictors_trans[:, j].reshape(-1, 1)).todense())))
            self.one_hot_encoders_nn.append(one_hot_encoder)
    else:
        predictors_qual_dummy = None

    initialize_neural_net(self, predictors_qual_dummy)

    full_hidden = concatenate(
        list(
            chain.from_iterable(
                [self.neural_net["liste_layers_quant_inputs"],
                 self.neural_net["liste_layers_qual_inputs"]])))
    output = Dense(1, activation='sigmoid')(full_hidden)
    self.model_nn = Model(
        inputs=list(chain.from_iterable([self.neural_net["liste_inputs_quant"],
                                         self.neural_net["liste_inputs_qual"]])),
        outputs=[output])

    # adam = tensorflow.keras.optimizers.Adam(lr={{choice([10 ** -3, 10 ** -2, 10 ** -1])}})
    # adam = tensorflow.keras.optimizers.Adam(lr=10 ** -2)
    # rmsprop = tensorflow.keras.optimizers.RMSprop(lr={{choice([10 ** -3, 10 ** -2, 10 ** -1])}})
    rmsprop = tensorflow.keras.optimizers.RMSprop(lr=0.5, rho=0.9, epsilon=None, decay=0.0)
    # sgd = tensorflow.keras.optimizers.SGD(lr={{choice([10 ** -3, 10 ** -2, 10 ** -1])}})

    # choiceval = {{choice(['adam', 'sgd', 'rmsprop'])}}
    # if choiceval == 'adam':
    #     optim = adam
    # elif choiceval == 'rmsprop':
    #     optim = rmsprop
    # else:
    optim = rmsprop

    self.model_nn.compile(loss='binary_crossentropy', optimizer=optim, metrics=['accuracy'])

    history = LossHistory(d_cont=self.d_cont, d_qual=self.d_qual, neural_net=self.neural_net)

    self.callbacks = [
        ReduceLROnPlateau(
            monitor='loss',
            factor=0.5,
            patience=10,
            verbose=0,
            mode='auto',
            min_delta=0.0001,
            cooldown=0,
            min_lr=0),
        history
    ]

    if self.predictors_cont is not None:
        if self.predictors_qual is not None:
            list_predictors = list(self.predictors_cont[self.train, :].T) + predictors_qual_dummy
        else:
            list_predictors = list(self.predictors_cont[self.train, :].T)
    else:
        if self.predictors_qual is not None:
            list_predictors = list(predictors_qual_dummy[self.train, :].T)
        else:
            logger.error("No training data provided.")

    self.model_nn.fit(
        list_predictors,
        self.labels[self.train],
        epochs=self.iter,
        batch_size=128,
        verbose=1,
        callbacks=self.callbacks
    )
