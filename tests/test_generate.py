#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import glmdisc

glmdisc = glmdisc.Glmdisc()


def test_init():
    assert glmdisc.test
    assert glmdisc.validation
    assert glmdisc.criterion == "bic"
    assert glmdisc.iter == 100
    assert glmdisc.m_start == 20


def test_generate():
    x, y = glmdisc.generate_data(800, 2, 1)
    assert x.shape[0] == 800
    assert x.shape[1] == 2
    assert len(x.shape) == 2
