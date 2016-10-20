"""
Module for testing util module.
"""

__author__ = 'wittawat'

import numpy as np
import matplotlib.pyplot as plt
import fsic.data as data
import fsic.feature as fea
import fsic.util as util
import fsic.kernel as kernel
import fsic.indtest as it
import fsic.glo as glo
import scipy.stats as stats

import unittest


class TestNumpySeedContext(unittest.TestCase):
    def setUp(self):
        pass 

    def test_context_deterministic(self):
        for s in [2, 98, 10]:
            with util.NumpySeedContext(seed=s):
                A1 = np.random.randn(5, 1)
                B1 = np.random.rand(6)

            C = np.random.rand(3)
            with util.NumpySeedContext(seed=s):
                A2 = np.random.randn(5, 1)
                B2 = np.random.rand(6)

            np.testing.assert_array_almost_equal(A1, A2)
            np.testing.assert_array_almost_equal(B1, B2)

    def tearDown(self):
        pass

class TestCCA(unittest.TestCase):
    def test_bounded(self):
        # between 0 and 1 
        n = 100
        for r in range(5):
            with util.NumpySeedContext(seed=r*5+1):
                X = np.random.randn(n, 2)*5 - 1
                Y = np.random.rand(n, 3) - 0.5
                evals, Vx, Vy = util.cca(X, Y, reg=1e-5)
                
                self.assertTrue(np.all(evals) <= 1)
                self.assertTrue(np.all(evals) >= -1)

class TestFunctions(unittest.TestCase):
    def test_bound_by_data(self):
        n, d = 50, 7
        m = n +3
        for s in [82, 22]:
            with util.NumpySeedContext(seed=s):
                Data = np.random.rand(n, d)
                Z = np.random.randn(m, d)*20
                P = util.bound_by_data(Z, Data)

                self.assertTrue(np.all(P.flatten() <= 1))
                self.assertTrue(np.all(P.flatten() >= 0))

                self.assertTrue(np.any(Z.flatten() > 1))
                self.assertTrue(np.any(Z.flatten() < 0))

                self.assertEqual(P.shape[0], Z.shape[0])
                self.assertEqual(P.shape[1], Z.shape[1])

    def test_one_of_K_code(self):
        arr = np.array([0, 1, 0, 2])
        Z = util.one_of_K_code(arr)
        exp = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1]])
        np.testing.assert_array_almost_equal(Z, exp)

                

