# the inclusion of the tests module is not meant to offer best practices for
# testing in general, but rather to support the `find_packages` example in
# setup.py that excludes installing the "tests" package

from unittest import TestCase

import glmdisc.glmdisc as glmdisc

class Testglmdisc(TestCase):

    x, y = glmdisc.generate_data(800, 2, 1)
    # Paramètres
    test = False
    validation = False
    criterion = 'bic'
    iter = 200
    m_depart = 10

    def test_is_string(self, x, y, test, validation, criterion, iter, m_depart):
        essai = glmdisc(x, y, iter, m_depart)

        #self.assertTrue(isinstance(s, basestring))
