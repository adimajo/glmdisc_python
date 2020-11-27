#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""discrete_data method for the glmdisc class.
"""
from loguru import logger

from glmdisc._discretizeDummy import discretize_dummy

CERTAINLY_OVERFIT_ = "N.B.: glmdisc most certainly overfit the training set!"

DISCRETIZED_TRAINING_SET_ = "Returning discretized training set."

MIGHT_OVERFIT_ = "N.B.: glmdisc might have overfit the validation set!"

DISCRETIZED_VALIDATION_SET_ = "Returning discretized validation set."

DISCRETIZED_TEST_SET_ = "Returning discretized test set."


def discrete_data(self):
    """
    Returns the best discrete data (train, validation or test) found by the MCMC.

    :rtype: numpy.ndarray
    """
    if self.predictors_cont is not None and self.predictors_qual is not None:
        if self.test:
            logger.info(DISCRETIZED_TEST_SET_)
            return discretize_dummy(self,
                                    self.predictors_cont[self.test_rows, :],
                                    self.predictors_qual[self.test_rows, :])
        if self.validation:
            logger.info(DISCRETIZED_VALIDATION_SET_)
            logger.info(MIGHT_OVERFIT_)
            return discretize_dummy(self,
                                    self.predictors_cont[self.validation_rows, :],
                                    self.predictors_qual[self.validation_rows, :])
        logger.info(DISCRETIZED_TRAINING_SET_)
        logger.info(CERTAINLY_OVERFIT_)
        return discretize_dummy(self,
                                self.predictors_cont[self.train_rows, :],
                                self.predictors_qual[self.train_rows, :])
    elif self.predictors_cont is not None:
        if self.test:
            logger.info(DISCRETIZED_TEST_SET_)
            return discretize_dummy(self,
                                    self.predictors_cont[self.test_rows, :],
                                    None)
        if self.validation:
            logger.info(DISCRETIZED_VALIDATION_SET_)
            logger.info(MIGHT_OVERFIT_)
            return discretize_dummy(self,
                                    self.predictors_cont[self.validation_rows, :],
                                    None)
        logger.info(DISCRETIZED_TRAINING_SET_)
        logger.info(CERTAINLY_OVERFIT_)
        return discretize_dummy(self,
                                self.predictors_cont[self.train_rows, :],
                                None)
    elif self.predictors_qual is not None:
        if self.test:
            logger.info(DISCRETIZED_TEST_SET_)
            return discretize_dummy(self,
                                    None,
                                    self.predictors_qual[self.test_rows, :])
        if self.validation:
            logger.info(DISCRETIZED_VALIDATION_SET_)
            logger.info(MIGHT_OVERFIT_)
            return discretize_dummy(self,
                                    None,
                                    self.predictors_qual[self.validation_rows, :])
        logger.info(DISCRETIZED_TRAINING_SET_)
        logger.info(CERTAINLY_OVERFIT_)
        return discretize_dummy(self,
                                None,
                                self.predictors_qual[self.train_rows, :])
