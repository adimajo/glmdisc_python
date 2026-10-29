#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

import glmdisc


def _scores(n=200, seed=0):
    rng = np.random.RandomState(seed)
    y = rng.binomial(1, 0.4, size=n)
    p = np.clip(0.3 * y + rng.uniform(size=n) * 0.7, 0, 1)
    return y, p


def test_gini_matches_auc():
    y, p = _scores()
    value, ci = glmdisc.gini(y, p)
    assert value == pytest.approx(2 * roc_auc_score(y, p) - 1)
    assert ci[0] < value < ci[1] <= 1


def test_gini_with_ties():
    y, p = _scores()
    p = np.round(p, 1)
    value, _ = glmdisc.gini(y, p)
    assert value == pytest.approx(2 * roc_auc_score(y, p) - 1)


def test_gini_with_sample_weight():
    y, p = _scores()
    w = np.random.RandomState(1).uniform(0.5, 2, size=len(y))
    value, _ = glmdisc.gini(y, p, sample_weight=w)
    assert value == pytest.approx(2 * roc_auc_score(y, p, sample_weight=w) - 1)
