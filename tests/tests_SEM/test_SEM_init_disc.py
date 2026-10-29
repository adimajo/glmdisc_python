#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.preprocessing import LabelEncoder

import glmdisc


def _model(n=300, levels=("a", "b", "c"), m_start=20):
    model = glmdisc.Glmdisc(m_start=m_start)
    model.predictors_cont = np.random.uniform(size=(n, 2))
    model.predictors_qual = np.random.choice(np.array(levels), size=(n, 2)).astype(object)
    model.labels = np.random.binomial(1, .5, size=n)
    return model


def test_init_disc_fits_each_label_encoder_once(monkeypatch):
    np.random.seed(0)
    model = _model()
    fits = []
    original_fit = LabelEncoder.fit

    def counting_fit(self, y):
        fits.append(len(y))
        return original_fit(self, y)

    monkeypatch.setattr(LabelEncoder, "fit", counting_fit)
    complete = model._calculate_shape()
    model._init_disc(complete)
    assert len(fits) == model.d_qual


def test_init_disc_draws_every_level():
    np.random.seed(0)
    model = _model(levels=("a", "b", "c"), m_start=20)
    edisc, predictors_trans = model._init_disc(model._calculate_shape())
    for j in range(model.d_qual):
        assert set(np.unique(edisc[:, model.d_cont + j])) == {0, 1, 2}


def test_init_disc_caps_groups_at_m_start():
    np.random.seed(0)
    model = _model(levels=tuple("abcdefgh"), m_start=3)
    edisc, predictors_trans = model._init_disc(model._calculate_shape())
    for j in range(model.d_qual):
        assert set(np.unique(edisc[:, model.d_cont + j])) == {0, 1, 2}


def test_init_disc_single_level():
    np.random.seed(0)
    model = _model(levels=("a",), m_start=20)
    edisc, predictors_trans = model._init_disc(model._calculate_shape())
    np.testing.assert_array_equal(edisc[:, 2:], 0)
