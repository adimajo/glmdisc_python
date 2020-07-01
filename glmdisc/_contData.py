#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cont_data method for glmdisc class.
"""


def cont_data(self):
    """
    Returns the continuous data provided to the MCMC.

    :rtype: list
    """
    return [self.predictors_cont, self.predictors_qual, self.labels]
