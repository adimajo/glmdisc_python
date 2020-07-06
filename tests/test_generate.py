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

    x, y, theta2 = glmdisc.Glmdisc.generate_data(n=800, d=d, theta=theta, seed=1)
    assert x.shape[0] == 800
    assert x.shape[1] == 2
    assert len(x.shape) == 2
    np.testing.assert_array_equal(theta, theta2)


def test_generate_theta_generated():
    d = 2

    x, y, theta = glmdisc.Glmdisc.generate_data(n=800, d=d, theta=None, seed=1)
    assert x.shape[0] == 800
    assert x.shape[1] == 2
    assert len(x.shape) == 2
    assert theta.shape == (3, 2)


def test_generate_theta_error():
    d = 2
    theta = "blabla"

    with pytest.raises(ValueError):
        glmdisc.Glmdisc.generate_data(n=800, d=d, theta=theta, seed=1)
