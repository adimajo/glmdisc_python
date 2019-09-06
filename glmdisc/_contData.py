#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 16:18:58 2019

@author: adrien
"""


def contData(self):
    """Returns the continuous data provided to the MCMC as
    a single pandas dataframe."""
    return [self.predictors_cont, self.predictors_qual, self.labels]
