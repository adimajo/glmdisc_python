#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fit function for NN algorithm
"""
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
import sklearn as sk
import sklearn.preprocessing
import tensorflow as tf
import tensorflow.keras.optimizers
from loguru import logger
from scipy import stats
from tensorflow.keras import Model
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import concatenate


def _plot_cont_soft_quant(self):
    for j in range(self.neural_net["d_cont"]):
        plt.xlim((np.nanmin(self.neural_net["predictors_cont"][:, j]),
                  (np.nanmax(self.neural_net["predictors_cont"][:, j]))))
        plt.ylim((0, 1))
        self.ax[j].clear()
        for k in range(self.best_outputs[j][0].shape[1]):
            self.ax[j].plot(np.sort(self.neural_net["predictors_cont"][:, j]),
                            self.best_outputs[j][0][np.argsort(
                                self.neural_net["predictors_cont"][:, j]), k],
                            color=['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'][(k + 2) % 8])
        self.fig.canvas.draw()
        plt.pause(0.05)


class LossHistory(Callback):
    """
    Custom Callback to evaluate current quantization scheme as a logistic regression
    """
    def __init__(self, d_cont, d_qual, neural_net):
        """
        Copying useful objects in self for function :code:`on_train_begin` and
        :code:`on_epoch_end` to which only self is provided.

        Parameters
        ----------
        d_cont: number of quantitative features
        d_qual: number of qualitative features
        neural_net: dictionary of neural net structure
        """
        super().__init__()
        self.plot_fit = neural_net['plot_fit']
        if self.plot_fit:
            self.fig, self.ax = plt.subplots(neural_net["d_cont"])
            self.fig.show()
            self.fig.canvas.draw()

        self.d_cont = d_cont
        self.d_qual = d_qual
        self.neural_net = neural_net
        self.current_weights = None
        self.best_weights = None
        self.losses = None
        self.best_criterion = None
        self.best_outputs = None
        self.best_encoders = None
        self.best_reglog = None

    def on_train_begin(self, logs=None):
        """
        Initializes losses and best_outputs as empty lists, best_criterion as infinite.

        Parameters
        ----------
        logs

        """
        self.losses = []
        self.best_criterion = float("inf")
        self.best_outputs = []

    def on_epoch_end(self, batch, logs=None):
        """
        Evaluates the proposed quantization scheme by going back to a "hard" quantization
        and fitting a logistic regression.

        .. todo:: parametrize burn in phase + don't bother evaluating disc for that phase

        .. todo:: plot qualitative?

        Parameters
        ----------
        batch
        logs
        """
        burn_in = 5
        self.current_weights = []
        for j in range(self.d_cont):
            self.current_weights.append(self.neural_net["liste_layers_quant"][j].get_weights())
        for j in range(self.d_qual):
            self.current_weights.append(self.neural_net["liste_layers_qual"][j].get_weights())
        performance, _, encoders, proposed_logistic_regression = _evaluate_disc(self, self.d_cont, self.d_qual, self.neural_net)
        self.losses.append(performance)
        if len(self.losses) > burn_in and self.losses[-1] < self.best_criterion:
            self.best_weights = []
            self.best_encoders = encoders
            self.best_outputs = []
            self.best_reglog = proposed_logistic_regression
            self.best_criterion = self.losses[-1]
            for j in range(self.d_cont):
                self.best_weights.append(self.current_weights[j])
                self.best_outputs.append(
                    tf.keras.backend.function([self.neural_net["liste_layers_quant"][j].input],
                                              [self.neural_net["liste_layers_quant"][j].output])(
                        [self.neural_net["predictors_cont"][:, j, np.newaxis]]))
            for j in range(self.d_qual):
                self.best_weights.append(self.current_weights[j + self.d_cont])
                self.best_outputs.append(
                    tf.keras.backend.function([self.neural_net["liste_layers_qual"][j].input],
                                              [self.neural_net["liste_layers_qual"][j].output])(
                        [self.neural_net["predictors_qual_dummy"][j]]))

            if self.plot_fit:
                _plot_cont_soft_quant(self)


def _initialize_neural_net(self, predictors_qual_dummy):
    """
    Constructs the neural network

    .. todo:: shouldn't predictors_qual_dummy be stored in self?
    """
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
        "d_cont": self.d_cont,
        "m_cont": m_cont,
        "d_qual": self.d_qual,
        "labels": self.labels,
        "predictors_cont": self.predictors_cont,
        "predictors_qual": self.predictors_qual,
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


def _from_layers_to_proba_training(d_cont, d_qual, neural_net):
    """
    Calculates the probability of each level for each feature from the provided neural net for the training set.

    Parameters
    ----------
    d_cont: number of quantitative features
    d_qual: number of qualitative features
    neural_net: neural net

    Returns
    -------
    list of size d_cont + d_qual of numpy.ndarray of probabilities for each level of each feature
    """
    results = [None] * (d_cont + d_qual)

    for j in range(d_cont):
        results[j] = tf.keras.backend.function([neural_net["liste_layers_quant"][j].input],
                                               [neural_net["liste_layers_quant"][j].output])(
            [neural_net["predictors_cont"][neural_net["train"], j, np.newaxis]])

    for j in range(d_qual):
        results[j + d_cont] = tf.keras.backend.function([neural_net["liste_layers_qual"][j].input],
                                                        [neural_net["liste_layers_qual"][j].output])(
            [neural_net["predictors_qual_dummy"][j][neural_net["train"]]])

    return results


def _from_weights_to_proba_test(d_cont, d_qual, m_cont, history, x_quant_test, x_qual_test, n_test, when="test"):
    """
    Calculates the probability of each level for each feature from the provided neural net for a test set.

    Parameters
    ----------
    type: during training or afterwards
    d_cont: number of quantitative features
    d_qual: number of qualitative features
    m_cont
    history
    x_quant_test
    x_qual_test
    n_test: number of test samples

    Returns
    -------
    list of size d_cont + d_qual of numpy.ndarray of probabilities for each level of each feature
    """
    if when == "test":
        the_weights = history.best_weights
    else:
        the_weights = history.current_weights

    results = [None] * (d_cont + d_qual)

    for j in range(d_cont):
        results[j] = np.zeros((n_test, m_cont[j]))
        for i in range(m_cont[j]):
            results[j][:, i] = the_weights[j][1][i] + the_weights[j][0][0][i] * x_quant_test[:, j]

    for j in range(d_qual):
        results[j + d_cont] = np.zeros((n_test, the_weights[j + d_cont][0].shape[1]))
        for i in range(the_weights[j + d_cont][0].shape[1]):
            for k in range(n_test):
                results[j + d_cont][k, i] = the_weights[j + d_cont][0][int(x_qual_test[k, j]), i]

    return results


def _evaluate_disc(history, d_cont, d_qual, neural_net):
    labels_idx = neural_net['train']
    proba = _from_layers_to_proba_training(d_cont, d_qual, neural_net)
    encoders = []
    X_transformed = np.ones((len(labels_idx), 1))
    results = [None] * (d_cont + d_qual)
    for j in range(d_cont + d_qual):
        results[j] = np.argmax(proba[j][0], axis=1)
        encoders.append(sk.preprocessing.OneHotEncoder(categories='auto', sparse=False, handle_unknown="ignore"))
        X_transformed = np.concatenate(
            (X_transformed,
             encoders[j].fit_transform(
                 X=results[j].reshape(-1, 1))),
            axis=1)

    proposed_logistic_regression = sk.linear_model.LogisticRegression(
        fit_intercept=False, solver="lbfgs", C=1e20, tol=1e-8, max_iter=50)

    proposed_logistic_regression.fit(X=X_transformed, y=neural_net["labels"][labels_idx].reshape(
        (labels_idx.shape[0],)))

    if neural_net['validation']:
        labels_idx = neural_net['validate']
        if neural_net["predictors_cont"] is not None:
            x_quant_test = neural_net["predictors_cont"][labels_idx]
        else:
            x_quant_test = None
        if neural_net["predictors_qual"] is not None:
            x_qual_test = neural_net["predictors_qual"][labels_idx]
        else:
            x_qual_test = None
        proba = _from_weights_to_proba_test(when="train",
                                            d_cont=d_cont,
                                            d_qual=d_qual,
                                            m_cont=neural_net["m_cont"],
                                            history=history,
                                            x_quant_test=x_quant_test,
                                            x_qual_test=x_qual_test,
                                            n_test=neural_net['validate'].shape[0])
        X_transformed = np.ones((len(neural_net['validate']), 1))
        for j in range(d_cont + d_qual):
            results[j] = np.argmax(proba[j], axis=1)
            X_transformed = np.concatenate(
                (X_transformed,
                 encoders[j].transform(
                     X=results[j].reshape(-1, 1))),
                axis=1)

    if neural_net["criterion"] in ['aic', 'bic']:
        loglik = sk.metrics.log_loss(
            neural_net["labels"][labels_idx],
            proposed_logistic_regression.predict_proba(X=X_transformed)[:, 1],
            normalize=False
        )
        if neural_net["validation"]:
            performance = 2 * loglik
            print("\n")
            logger.info("Current likelihood on validation set: " + str(performance / 2.0))
        elif neural_net["criterion"] == "bic":
            performance = 2 * loglik + proposed_logistic_regression.coef_.shape[1] * np.log(
                labels_idx.shape[0])
            print("\n")
            logger.info("Current BIC on training set: " + str(- performance))
        else:
            performance = 2 * loglik + 2 * proposed_logistic_regression.coef_.shape[1]
            print("\n")
            logger.info("Current AIC on training set: " + str(- performance))
    else:
        performance = sk.metrics.roc_auc_score(
            y_true=neural_net['labels'][labels_idx],
            y_score=proposed_logistic_regression.predict_proba(X=X_transformed)[:, 1])
        print("\n")
        logger.info("Current Gini on training set: " + str(performance))

    predicted = proposed_logistic_regression.predict_proba(X_transformed)[:, 1]

    return performance, predicted, encoders, proposed_logistic_regression


def _prepare_inputs(self, predictors_trans):
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

    if self.predictors_cont is not None:
        if self.predictors_qual is not None:
            list_predictors = list(self.predictors_cont[self.train, :].T) + \
                              [x[self.train, :] for x in predictors_qual_dummy]
        else:
            list_predictors = list(self.predictors_cont[self.train, :].T)
    else:
        if self.predictors_qual is not None:
            list_predictors = [x[self.train, :] for x in predictors_qual_dummy]
        else:
            msg = "No training data provided."
            logger.error(msg)
            raise ValueError(msg)

    return predictors_qual_dummy, list_predictors


def _compile_and_fit_neural_net(self, optim, list_predictors):

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

    self.model_nn.compile(loss='binary_crossentropy', optimizer=optim, metrics=['accuracy'])

    self.model_nn.fit(
        list_predictors,
        self.labels[self.train],
        epochs=self.iter,
        batch_size=128,
        verbose=1,
        callbacks=self.callbacks
    )


def _parse_kwargs(self, **kwargs):
    if 'plot' in kwargs:
        if not isinstance(kwargs['plot'], bool):
            msg = "plot parameter provided but not boolean"
            logger.error(msg)
            raise ValueError(msg)
        else:
            self.neural_net['plot_fit'] = kwargs['plot']
    else:
        self.neural_net['plot_fit'] = False

    if "optimizer" in kwargs:
        optim = kwargs['optimizer']
    else:
        optim = tensorflow.keras.optimizers.RMSprop(lr=0.5, rho=0.9, epsilon=None, decay=0.0)

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
        LossHistory(d_cont=self.d_cont,
                    d_qual=self.d_qual,
                    neural_net=self.neural_net)]

    if "callbacks" in kwargs:
        self.callbacks.append(kwargs['callbacks'])

    return optim


def _fit_nn(self, predictors_trans, **kwargs):
    predictors_qual_dummy, list_predictors = _prepare_inputs(self=self, predictors_trans=predictors_trans)

    _initialize_neural_net(self=self, predictors_qual_dummy=predictors_qual_dummy)

    optim = _parse_kwargs(self=self, **kwargs)

    _compile_and_fit_neural_net(self=self, optim=optim, list_predictors=list_predictors)

    self.best_reglog = self.callbacks[1].best_reglog
