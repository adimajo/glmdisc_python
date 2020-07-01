#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
discrete_data method for the glmdisc class.
"""
from glmdisc._discretizeDummy import discretize_dummy


def discrete_data(self):
    """
    Returns the best discrete data found by the MCMC.

    .. todo:: specify rtype
    """
    return discretize_dummy(self, self.predictors_cont[self.splitting[-1], ],
                            self.predictors_qual[self.splitting[-1], ])
