#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
discreteData method for the glmdisc class.
"""


def discrete_data(self):
    """
    Returns the best discrete data found by the MCMC.
    """
    return discretizeDummy(self, self.predictors_cont[self.splitting[-1], ],
                           self.predictors_qual[self.splitting[-1], ])
