"""
Module for testing data module.
"""

__author__ = 'wittawat'

import numpy as np
import matplotlib.pyplot as plt
import fsic.data as data
import fsic.util as util
import fsic.kernel as kernel
import fsic.indtest as it
import fsic.glo as glo
import scipy.stats as stats

import unittest

class TestPairedData(unittest.TestCase):
    def setUp(self):
        pass 

    def test_add(self):
        n1 = 30
        n2 = 20
        dx = 2
        dy = 1

        X1 = np.random.randn(n1, dx)
        Y1 = np.random.rand(n1, dy)
        X2 = np.random.rand(n2, dx)
        Y2 = np.random.randn(n2, dy) + 1

        pdata1 = data.PairedData(X1, Y1)
        pdata2 = data.PairedData(X2, Y2)
        # merge
        pdata = pdata1 + pdata2

        # check internals 
        X = pdata.X 
        Y = pdata.Y
        np.testing.assert_array_almost_equal(X[:n1], X1)
        np.testing.assert_array_almost_equal(X[n1:], X2)
        np.testing.assert_array_almost_equal(Y[:n1], Y1)
        np.testing.assert_array_almost_equal(Y[n1:], Y2)
        self.assertTrue(pdata != pdata1)
        self.assertTrue(pdata != pdata2)
        # test size
        self.assertEqual(pdata.sample_size(), n1+n2)
        self.assertEqual(pdata1.sample_size(), n1)
        self.assertEqual(pdata2.sample_size(), n2)


    def tearDown(self):
        pass

# end class TestPairedData

class TestPSStraResample(unittest.TestCase):
    def test_sample(self):
        import math

        for s in [27, 91]:
            n_ori = 200
            p_fracs = [0.1, 0.5, 0.4]
            X = np.random.randn(n_ori, 3)
            Y = np.array([0]*int(p_fracs[0]*n_ori) + [1]*int(p_fracs[1]*n_ori) +
                    [2]*int(p_fracs[2]*n_ori) )[:, np.newaxis]
            pdata_ori = data.PairedData(X, Y)
            ps = data.PSStraResample(pdata_ori, Y[:, 0])

            m = 79
            pdata = ps.sample(m, seed=s)
            self.assertEqual(pdata.sample_size(), m)

            x, y = pdata.xy()
            yu, counts = np.unique(y, return_counts=True)
            for i, u in enumerate(yu):
                self.assertTrue( counts[i] - int(p_fracs[i]*m) <= 1 )



class TestPSNullResample(unittest.TestCase):
    def test_sample_deterministic(self):
        seeds = [2, 98, 72]
        for s in seeds:
            n = 21
            pdata = data.PSUnifRotateNoise(angle=np.pi/3, noise_dim=1).sample(n, seed=s)
            null_ps = data.PSNullResample(pdata)

            m = n/2
            shuff1 = null_ps.sample(m, seed=s+1)
            shuff2 = null_ps.sample(m, seed=s+1)

            X1, Y1 = shuff1.xy()
            X2, Y2 = shuff2.xy()
            np.testing.assert_array_almost_equal(X1, X2)
            np.testing.assert_array_almost_equal(Y1, Y2)

class TestPSGaussNoiseDim(unittest.TestCase):
    def test_sample(self):
        ndxs = [0, 2, 3]
        ndys = [3, 0, 4]
        ori_ps = data.PS2DUnifRotate(np.pi/3)
        n = 10
        for i, ndx, ndy in zip(list(range(len(ndxs))), ndxs, ndys):
            ps = data.PSGaussNoiseDims(ori_ps, ndx, ndy)
            pdata = ps.sample(n=n, seed=83)
            X, Y = pdata.xy()

            self.assertEqual(X.shape[0], n)
            self.assertEqual(Y.shape[0], n)
            self.assertEqual(X.shape[1], ori_ps.dx()+ndx)
            self.assertEqual(Y.shape[1], ori_ps.dy()+ndy)

            self.assertTrue(np.all(np.isfinite(X)))
            self.assertTrue(np.all(np.isfinite(Y)))



class TestPS2DSinFreq(unittest.TestCase):
    def setUp(self):
        pass 

    def test_sample(self):
        ps = data.PS2DSinFreq(1)
        for n in [5, 613]:
            pdata = ps.sample(n=n)
            X, Y = pdata.xy()
            XY = np.hstack((X, Y))

            self.assertEqual(X.shape[1], 1)
            self.assertEqual(Y.shape[1], 1)
            self.assertEqual(XY.shape[0], n)

    def tearDown(self):
        pass

# end class TestPS2DSinFreq

class TestPSPairwiseSign(unittest.TestCase): 
    def setUp(self):
        pass 

    def test_dim(self):
        n = 10
        for d in [2, 4]:
            ps = data.PSPairwiseSign(dx=d)
            pdata = ps.sample(n=n, seed=d+1)
            X, Y = pdata.xy()
            
            self.assertEqual(X.shape[0], n)
            self.assertEqual(Y.shape[0], n)
            self.assertEqual(X.shape[1], d)
            self.assertEqual(Y.shape[1], 1)
            

    def tearDown(self):
        pass
