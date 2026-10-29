#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pytest
import numpy as np
import glmdisc


def test_generate_theta_provided():
    d = 2
    theta = np.array([[1] * d] * 3)
    theta[1, :] = 2
    theta[2, :] = -2

    x, y, theta2 = glmdisc.Glmdisc.generate_data(n=800, d=d, theta=theta)
    assert x.shape[0] == 800
    assert x.shape[1] == 2
    assert len(x.shape) == 2
    np.testing.assert_array_equal(theta, theta2)


def test_generate_theta_generated():
    d = 2

    x, y, theta = glmdisc.Glmdisc.generate_data(n=800, d=d, theta=None)
    assert x.shape[0] == 800
    assert x.shape[1] == 2
    assert len(x.shape) == 2
    assert theta.shape == (3, 2)


def test_generate_theta_error():
    d = 2
    theta = "blabla"

    with pytest.raises(ValueError):
        glmdisc.Glmdisc.generate_data(n=800, d=d, theta=theta)


def test_generate_log_odd_not_truncated(monkeypatch):
    theta = np.array([[0.4], [0.9], [-0.6]])
    drawn = {}

    def fake_binomial(n, p, size=None):
        drawn["p"] = p
        return np.zeros(len(p), dtype=int)

    monkeypatch.setattr(np.random, "binomial", fake_binomial)
    np.random.seed(0)
    x, y, theta2 = glmdisc.Glmdisc.generate_data(n=50, d=1, theta=theta)

    levels = np.digitize(x[:, 0], [0.333, 0.666])
    expected = 1 / (1 + np.exp(-theta[levels, 0]))
    np.testing.assert_allclose(drawn["p"], expected)


def test_generate_labels_size(monkeypatch):
    sizes = []
    binomial = np.random.binomial

    def spy_binomial(n, p, size=None):
        sizes.append(size)
        return binomial(n, p, size=size)

    monkeypatch.setattr(np.random, "binomial", spy_binomial)
    x, y, theta = glmdisc.Glmdisc.generate_data(n=1, d=3)
    assert sizes == [1]
    assert y.shape == (1,)
