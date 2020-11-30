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
    """
    Plots the soft quantization while fitting
    """
    for j in range(self.glmdisc_object.d_cont):
        plt.xlim((np.nanmin(self.glmdisc_object.predictors_cont[:, j]),
                  (np.nanmax(self.glmdisc_object.predictors_cont[:, j]))))
        plt.ylim((0, 1))
        self.ax[j].clear()
        for k in range(self.best_outputs[j][0].shape[1]):
            self.ax[j].plot(np.sort(self.glmdisc_object.predictors_cont[:, j]),
                            self.best_outputs[j][0][np.argsort(
                                self.glmdisc_object.predictors_cont[:, j]), k],
                            color=['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'][(k + 2) % 8])
        self.fig.canvas.draw()
        plt.pause(0.05)


class LossHistory(Callback):
    """
    Custom Callback to evaluate current quantization scheme as a logistic regression
    """
    def __init__(self, d_cont, d_qual, glmdisc_object):
        """
        Copying useful objects in self for function :code:`on_train_begin` and
        :code:`on_epoch_end` to which only self is provided.

        Parameters
        ----------
        d_cont: number of quantitative features
        d_qual: number of qualitative features
        glmdisc_object: model
        """
        super().__init__()
        self.plot_fit = glmdisc_object.plot_fit
        self.burn_in = glmdisc_object.burn_in
        if self.plot_fit:
            self.fig, self.ax = plt.subplots(glmdisc_object.d_cont)
            self.fig.show()
            self.fig.canvas.draw()

        self.d_cont = d_cont
        self.d_qual = d_qual
        self.glmdisc_object = glmdisc_object
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

    def on_train_end(self, logs=None):
        """
        Initializes losses and best_outputs as empty lists, best_criterion as infinite.

        Parameters
        ----------
        logs

        """
        del self.glmdisc_object

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
        self.current_weights = []
        for j in range(self.d_cont):
            self.current_weights.append(self.glmdisc_object.model_nn["liste_layers_quant"][j].get_weights())
        for j in range(self.d_qual):
            self.current_weights.append(self.glmdisc_object.model_nn["liste_layers_qual"][j].get_weights())
        performance, _, encoders, proposed_logistic_regression = _evaluate_disc(self,
                                                                                self.d_cont,
                                                                                self.d_qual,
                                                                                self.glmdisc_object)
        self.losses.append(performance)
        if len(self.losses) > self.burn_in and self.losses[-1] < self.best_criterion:
            self.best_weights = []
            self.best_encoders = encoders
            self.best_outputs = []
            self.best_reglog = proposed_logistic_regression
            self.best_criterion = self.losses[-1]
            for j in range(self.d_cont):
                self.best_weights.append(self.current_weights[j])
                self.best_outputs.append(
                    tf.keras.backend.function([self.glmdisc_object.model_nn["liste_layers_quant"][j].input],
                                              [self.glmdisc_object.model_nn["liste_layers_quant"][j].output])(
                        [self.glmdisc_object.predictors_cont[:, j, np.newaxis]]))
            for j in range(self.d_qual):
                self.best_weights.append(self.current_weights[j + self.d_cont])
                self.best_outputs.append(
                    tf.keras.backend.function([self.glmdisc_object.model_nn["liste_layers_qual"][j].input],
                                              [self.glmdisc_object.model_nn["liste_layers_qual"][j].output])(
                        [self.glmdisc_object.predictors_qual_dummy[j]]))

            if self.plot_fit:
                _plot_cont_soft_quant(self)


def _initialize_neural_net(self):
    """
    Constructs the neural network by putting TensorFlow objects :code:`Inputs` and layers in a dictionary stored in
    the class object.
    """
    m_cont = [self.m_start] * self.d_cont
    m_qual = [self.m_start] * (self.d_cont + self.d_qual)

    for j in range(self.d_qual):
        num_levels = stats.describe(sk.preprocessing.LabelEncoder().fit(self.predictors_qual[:, j]).transform(
            self.predictors_qual[:, j])).minmax[1]
        if self.m_start > num_levels:
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
        liste_layers_qual[i] = Dense(
            m_qual[i],
            activation='softmax',
            use_bias=False)

        liste_layers_qual_inputs[i] = liste_layers_qual[i](
            liste_inputs_qual[i])

    self.model_nn.update({
        "liste_inputs_quant": liste_inputs_quant,
        "liste_layers_quant": liste_layers_quant,
        "liste_layers_quant_inputs": liste_layers_quant_inputs,
        "liste_inputs_qual": liste_inputs_qual,
        "liste_layers_qual": liste_layers_qual,
        "liste_layers_qual_inputs": liste_layers_qual_inputs
    })


def _from_layers_to_proba_training(d_cont, d_qual, glmdisc_object):
    """
    Calculates the probability of each level for each feature from the provided neural net for the training set.

    Parameters
    ----------
    d_cont: number of quantitative features
    d_qual: number of qualitative features
    glmdisc_object: model

    Returns
    -------
    list of size d_cont + d_qual of numpy.ndarray of probabilities for each level of each feature
    """
    results = [None] * (d_cont + d_qual)

    for j in range(d_cont):
        results[j] = tf.keras.backend.function([glmdisc_object.model_nn["liste_layers_quant"][j].input],
                                               [glmdisc_object.model_nn["liste_layers_quant"][j].output])(
            [glmdisc_object.predictors_cont[glmdisc_object.train_rows, j, np.newaxis]])

    for j in range(d_qual):
        results[j + d_cont] = tf.keras.backend.function([glmdisc_object.model_nn["liste_layers_qual"][j].input],
                                                        [glmdisc_object.model_nn["liste_layers_qual"][j].output])(
            [glmdisc_object.predictors_qual_dummy[j][glmdisc_object.train_rows]])

    return results


def _from_weights_to_proba_test(d_cont, d_qual, m_cont, history, x_quant_test, x_qual_test, n_test, when="test"):
    """
    Calculates the probability of each level for each feature from the provided neural net for a test set.

    Parameters
    ----------
    d_cont: number of quantitative features
    d_qual: number of qualitative features
    m_cont: number of levels for continuous features
    history: the custom Callback's history
    x_quant_test: the continuous features to discretize
    x_qual_test: the qualitative features which levels are grouped
    n_test: number of test samples
    when: if "test", then apply the :code:`best_weights`, otherwise the :code:`current_weights`

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


def _evaluate_disc(history, d_cont: int, d_qual: int, glmdisc_object):
    """
    Evaluates the quality of a proposed quantization

    Parameters
    ----------
    history: the custom Callback's history
    d_cont: number of continuous features
    d_qual: number of categorical features
    glmdisc_object

    Returns
    ----------
    performance: the performance (depends on the provided criterion) of the current quantization
    predicted: the probability of class 1 for each training or validation (if :code:`validation` is True) samples
    encoders: the one hot encoders used for this quantization
    proposed_logistic_regression: the logistic regression associated with this quantization (always on training set)
    """
    labels_idx = glmdisc_object.train_rows
    proba = _from_layers_to_proba_training(d_cont, d_qual, glmdisc_object)
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

    proposed_logistic_regression.fit(X=X_transformed, y=glmdisc_object.labels[labels_idx].reshape(
        (labels_idx.shape[0],)))

    if glmdisc_object.validation:
        labels_idx = glmdisc_object.validation_rows
        if glmdisc_object.predictors_cont is not None:
            x_quant_test = glmdisc_object.predictors_cont[labels_idx]
        else:
            x_quant_test = None
        if glmdisc_object.predictors_qual is not None:
            x_qual_test = glmdisc_object.predictors_qual[labels_idx]
        else:
            x_qual_test = None
        proba = _from_weights_to_proba_test(when="train",
                                            d_cont=d_cont,
                                            d_qual=d_qual,
                                            m_cont=[glmdisc_object.m_start] * glmdisc_object.d_cont,
                                            history=history,
                                            x_quant_test=x_quant_test,
                                            x_qual_test=x_qual_test,
                                            n_test=glmdisc_object.validation_rows.shape[0])
        X_transformed = np.ones((len(glmdisc_object.validation_rows), 1))
        for j in range(d_cont + d_qual):
            results[j] = np.argmax(proba[j], axis=1)
            X_transformed = np.concatenate(
                (X_transformed,
                 encoders[j].transform(
                     X=results[j].reshape(-1, 1))),
                axis=1)

    if glmdisc_object.criterion in ['aic', 'bic']:
        loglik = sk.metrics.log_loss(
            glmdisc_object.labels[labels_idx],
            proposed_logistic_regression.predict_proba(X=X_transformed)[:, 1],
            normalize=False
        )
        if glmdisc_object.validation:
            performance = 2 * loglik
            print("\n")
            logger.info("Current likelihood on validation set: " + str(performance / 2.0))
        elif glmdisc_object.criterion == "bic":
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
            y_true=glmdisc_object.labels[labels_idx],
            y_score=proposed_logistic_regression.predict_proba(X=X_transformed)[:, 1])
        print("\n")
        logger.info("Current Gini on training set: " + str(performance))

    predicted = proposed_logistic_regression.predict_proba(X_transformed)[:, 1]

    return performance, predicted, encoders, proposed_logistic_regression


def _prepare_inputs(self, predictors_trans):
    """
    Transforms categorical inputs into dummies and prepares inputs into an appropriate list for Tensorflow

    Returns
    _______
    list_predictors: list of np.ndarray
    """
    if self.predictors_qual is not None:
        self.model_nn['one_hot_encoders_nn'] = []
        self.predictors_qual_dummy = []
        for j in range(self.d_qual):
            one_hot_encoder = sk.preprocessing.OneHotEncoder()
            self.predictors_qual_dummy.append(np.squeeze(np.asarray(
                one_hot_encoder.fit_transform(predictors_trans[:, j].reshape(-1, 1)).todense())))
            self.model_nn['one_hot_encoders_nn'].append(one_hot_encoder)
    else:
        self.predictors_qual_dummy = None

    if self.predictors_cont is not None:
        if self.predictors_qual is not None:
            list_predictors = list(self.predictors_cont[self.train_rows, :].T) + \
                              [x[self.train_rows, :] for x in self.predictors_qual_dummy]
        else:
            list_predictors = list(self.predictors_cont[self.train_rows, :].T)
    elif self.predictors_qual is not None:
        list_predictors = [x[self.train_rows, :] for x in self.predictors_qual_dummy]

    return list_predictors


def _compile_and_fit_neural_net(self, optim, list_predictors):
    """
    Creates, compiles and fits the Tensorflow model
    """
    full_hidden = concatenate(
        list(
            chain.from_iterable(
                [self.model_nn["liste_layers_quant_inputs"],
                 self.model_nn["liste_layers_qual_inputs"]])))
    output = Dense(1, activation='sigmoid')(full_hidden)
    self.model_nn["tensorflow_model"] = Model(
        inputs=list(chain.from_iterable([self.model_nn["liste_inputs_quant"],
                                         self.model_nn["liste_inputs_qual"]])),
        outputs=[output])

    self.model_nn["tensorflow_model"].compile(loss='binary_crossentropy', optimizer=optim, metrics=['accuracy'])

    self.model_nn["tensorflow_model"].fit(
        list_predictors,
        self.labels[self.train_rows],
        epochs=self.iter,
        batch_size=128,
        verbose=1,
        callbacks=self.model_nn["callbacks"]
    )


def _parse_kwargs(self, **kwargs):
    """
    Parses eventual kwargs to the :code:`fit` method, like plot, optimizer and callbacks.
    """
    if 'plot' in kwargs:
        if not isinstance(kwargs['plot'], bool):
            msg = "plot parameter provided but not boolean"
            logger.error(msg)
            raise ValueError(msg)
        else:
            self.plot_fit = kwargs['plot']

    if "optimizer" in kwargs:
        optim = kwargs['optimizer']
    else:
        optim = tensorflow.keras.optimizers.RMSprop(lr=0.5, rho=0.9, epsilon=None, decay=0.0)

    self.model_nn["callbacks"] = [
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
                    glmdisc_object=self)]

    if "callbacks" in kwargs:
        self.model_nn["callbacks"].append(kwargs['callbacks'])

    return optim


def _final_result(self):
    if self.test:
        if self.predictors_cont is None:
            pred_cont = None
        else:
            pred_cont = self.predictors_cont[self.test_rows, :]
        if self.predictors_qual is None:
            pred_qual = None
        else:
            pred_qual = self.predictors_qual[self.test_rows, :]
        subset = "test"
        labels = self.labels[self.test_rows]
        predictions = self.predict(predictors_cont=pred_cont, predictors_qual=pred_qual)
    elif self.validation:
        if self.predictors_cont is None:
            pred_cont = None
        else:
            pred_cont = self.predictors_cont[self.validation_rows, :]
        if self.predictors_qual is None:
            pred_qual = None
        else:
            pred_qual = self.predictors_qual[self.validation_rows, :]
        subset = "validation"
        labels = self.labels[self.validation_rows]
        predictions = self.predict(predictors_cont=pred_cont, predictors_qual=pred_qual)
    else:
        if self.predictors_cont is None:
            pred_cont = None
        else:
            pred_cont = self.predictors_cont[self.train_rows, :]
        if self.predictors_qual is None:
            pred_qual = None
        else:
            pred_qual = self.predictors_qual[self.train_rows, :]
        subset = "training"
        labels = self.labels[self.train_rows]
        predictions = self.predict(predictors_cont=pred_cont, predictors_qual=pred_qual)

    if self.criterion in ['aic', 'bic']:
        loglik = sk.metrics.log_loss(
            labels,
            predictions[:, 1],
            normalize=False
        )
        if self.validation | self.test:
            performance = 2 * loglik
            print("\n")
            logger.info("Best likelihood on " + subset + " set: " + str(performance / 2.0))
        elif self.criterion == "bic":
            performance = 2 * loglik + self.best_reglog.coef_.shape[1] * np.log(
                self.train_rows.shape[0])
            print("\n")
            logger.info("Best BIC on training set: " + str(- performance))
        else:
            performance = 2 * loglik + 2 * self.best_reglog.coef_.shape[1]
            print("\n")
            logger.info("Best AIC on training set: " + str(- performance))
    else:
        performance = sk.metrics.roc_auc_score(
            y_true=labels,
            y_score=predictions[:, 1])
        print("\n")
        logger.info("Best Gini on " + subset + " set: " + str(performance))


def _fit_nn(self, predictors_trans, **kwargs):
    """
    Wrap-up method: calls all functions to prepare inputs, construct the layers, parse the kwargs, creating, compiling
    and fitting the neural network, then stores the best results.
    """
    list_predictors = _prepare_inputs(self=self, predictors_trans=predictors_trans)

    _initialize_neural_net(self=self)

    optim = _parse_kwargs(self=self, **kwargs)

    _compile_and_fit_neural_net(self=self, optim=optim, list_predictors=list_predictors)

    self.best_reglog = self.model_nn["callbacks"][1].best_reglog
    self.performance = self.model_nn["callbacks"][1].best_criterion
    self.criterion_iter = self.model_nn["callbacks"][1].losses

    _final_result(self)
