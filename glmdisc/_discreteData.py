#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""discrete_data method for the glmdisc class.
"""
from glmdisc._discretizeDummy import discretize_dummy
from loguru import logger

CERTAINLY_OVERFIT_ = "N.B.: glmdisc most certainly overfit the training set!"

DISCRETIZED_TRAINING_SET_ = "Returning discretized training set."

MIGHT_OVERFIT_ = "N.B.: glmdisc might have overfit the validation set!"

DISCRETIZED_VALIDATION_SET_ = "Returning discretized validation set."

DISCRETIZED_TEST_SET_ = "Returning discretized test set."


def discrete_data(self):
    """
    Returns the best discrete data (train, validation or test) found by the MCMC.

    :rtype: numpy.array
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
                                    self.predictors_cont[self.validate, :],
                                    self.predictors_qual[self.validate, :])
        logger.info(DISCRETIZED_TRAINING_SET_)
        logger.info(CERTAINLY_OVERFIT_)
        return discretize_dummy(self,
                                self.predictors_cont[self.train, :],
                                self.predictors_qual[self.train, :])
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
                                    self.predictors_cont[self.validate, :],
                                    None)
        logger.info(DISCRETIZED_TRAINING_SET_)
        logger.info(CERTAINLY_OVERFIT_)
        return discretize_dummy(self,
                                self.predictors_cont[self.train, :],
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
                                    self.predictors_qual[self.validate, :])
        logger.info(DISCRETIZED_TRAINING_SET_)
        logger.info(CERTAINLY_OVERFIT_)
        return discretize_dummy(self,
                                None,
                                self.predictors_qual[self.train, :])
