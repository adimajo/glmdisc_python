#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import glmdisc


def test_generate():
    d = 2
    theta = np.array([[1] * d] * 3)
    theta[1, :] = 2
    theta[2, :] = -2

    x, y, theta2 = glmdisc.Glmdisc.generate_data(n=800, d=d, theta=theta, seed=1)
    assert x.shape[0] == 800
    assert x.shape[1] == 2
    assert len(x.shape) == 2
    np.testing.assert_array_equal(theta, theta2)
