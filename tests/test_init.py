#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pytest
import numpy as np
import glmdisc


def test_init():
    glmdisc_instance = glmdisc.Glmdisc()
    assert glmdisc_instance.test
    assert glmdisc_instance.validation
    assert glmdisc_instance.criterion == "bic"
    assert glmdisc_instance.iter == 100
    assert glmdisc_instance.m_start == 20
    assert glmdisc_instance.criterion_iter == []
    assert glmdisc_instance.best_link == []
    assert glmdisc_instance.best_reglog is None
    assert glmdisc_instance.affectations == []
    assert glmdisc_instance.best_encoder_emap is None
    assert glmdisc_instance.performance == -np.inf
    np.testing.assert_array_equal(glmdisc_instance.train, np.array([]))
    np.testing.assert_array_equal(glmdisc_instance.validate, np.array([]))
    np.testing.assert_array_equal(glmdisc_instance.test_rows, np.array([]))


def test_criterion():
    glmdisc.Glmdisc(criterion="bic")
    glmdisc.Glmdisc(criterion="aic")
    glmdisc.Glmdisc(criterion="gini")

    with pytest.raises(ValueError):
        glmdisc.Glmdisc(criterion="toto")


def test_test():
    glmdisc.Glmdisc(test=True)
    glmdisc.Glmdisc(test=False)

    with pytest.raises(ValueError):
        glmdisc.Glmdisc(test="some string")

    with pytest.raises(ValueError):
        glmdisc.Glmdisc(test=12)


def test_validation():
    glmdisc.Glmdisc(validation=True)
    glmdisc.Glmdisc(validation=False)

    with pytest.raises(ValueError):
        glmdisc.Glmdisc(validation="some string")

    with pytest.raises(ValueError):
        glmdisc.Glmdisc(validation=12)


def test_iter():
    glmdisc.Glmdisc(iter=100)

    with pytest.raises(ValueError):
        glmdisc.Glmdisc(iter=-12)

    with pytest.raises(ValueError):
        glmdisc.Glmdisc(iter=100000000)


def test_m_start():
    glmdisc.Glmdisc(m_start=10)

    with pytest.raises(ValueError):
        glmdisc.Glmdisc(m_start=1)


def test_gini_penalized(caplog):
    glmdisc.Glmdisc(validation=False,
                    criterion="gini")
    assert caplog.records[0].message == 'Using Gini index on training set might yield an overfitted model'


def test_validation_criterion(caplog):
    glmdisc.Glmdisc(validation=True,
                    criterion='aic')

    assert caplog.records[0].message == ('No need to penalize the log-likelihood when a validation set is used. '
                                         'Using log-likelihood instead.')

    glmdisc.Glmdisc(validation=True,
                    criterion='bic')

    assert caplog.records[0].message == ('No need to penalize the log-likelihood when a validation set is used. '
                                         'Using log-likelihood instead.')
