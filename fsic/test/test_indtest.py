"""
Module for testing indtest module.
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


def get_pdata_mean(n, dx=2):
    X = np.random.randn(n, dx)
    Y = np.mean(X, 1)[:, np.newaxis] + np.random.randn(n, 1)*0.01
    return data.PairedData(X, Y, label='mean')

def kl_median(pdata):
    """
    Get two Gaussian kernels constructed with the median heuristic.
    Randomize V, W from the standard Gaussian distribution.
    """
    xtr, ytr = pdata.xy()
    dx = xtr.shape[1]
    dy = ytr.shape[1]
    medx2 = util.meddistance(xtr)**2
    medy2 = util.meddistance(ytr)**2
    k = kernel.KGauss(medx2)
    l = kernel.KGauss(medy2)
    return k, l

class TestNFSIC(unittest.TestCase):
    def setUp(self):
        n = 300
        dx = 2
        pdata_mean = get_pdata_mean(n, dx)
        X, Y = pdata_mean.xy()
        gwx2 = util.meddistance(X)**2
        gwy2 = util.meddistance(Y)**2
        k = kernel.KGauss(gwx2)
        l = kernel.KGauss(gwy2)
        J = 2
        V = np.random.randn(J, dx)
        W = np.random.randn(J, 1)

        self.nfsic = it.NFSIC(k, l, V, W, alpha=0.01)
        self.pdata_mean = pdata_mean

    def test_perform_test(self):
        test_result = self.nfsic.perform_test(self.pdata_mean)
        # should reject. Cannot assert this for sure.
        #self.assertTrue(test_result['h0_rejected'], 'Test should reject H0')
        pass 

    def test_compute_stat(self):
        stat = self.nfsic.compute_stat(self.pdata_mean)
        self.assertGreater(stat, 0)

    def test_list_permute(self):
        # Check that the relative frequency in the simulated histogram is 
        # accurate enough. 
        ps = data.PS2DSinFreq(freq=2)
        n_permute = 1000
        J = 4
        for s in [284, 77]:
            with util.NumpySeedContext(seed=s):
                pdata = ps.sample(n=200, seed=s+1)
                dx = pdata.dx()
                dy = pdata.dy()
                X, Y = pdata.xy()

                k = kernel.KGauss(2)
                l = kernel.KGauss(3)
                V = np.random.randn(J, dx)
                W = np.random.randn(J, dy)
                #nfsic = it.NFSIC(k, l, V, W, alpha=0.01, reg=0, n_permute=n_permute,
                #        seed=s+3):

                #nfsic_result = nfsic.perform_test(pdata)
                arr = it.NFSIC.list_permute(X, Y, k, l, V, W, n_permute=n_permute,
                        seed=s+34, reg=0)
                arr_naive = it.NFSIC._list_permute_naive(X, Y, k, l, V, W,
                        n_permute=n_permute, seed=s+389, reg=0)
                

                # make sure that the relative frequency of the histogram does 
                # not differ much.
                freq_a, edge_a = np.histogram(arr)
                freq_n, edge_n = np.histogram(arr_naive)
                nfreq_a = freq_a/float(np.sum(freq_a))
                nfreq_n = freq_n/float(np.sum(freq_n))
                arr_diff = np.abs(nfreq_a - nfreq_n)
                self.assertTrue(np.all(arr_diff <= 0.2))


    def tearDown(self):
        pass

class TestGaussNFSIC(unittest.TestCase):
    def setUp(self):
        n = 300
        dx = 2
        pdata_mean = get_pdata_mean(n, dx)
        X, Y = pdata_mean.xy()
        gwx2 = util.meddistance(X)**2
        gwy2 = util.meddistance(Y)**2
        J = 2
        V = np.random.randn(J, dx)
        W = np.random.randn(J, 1)

        self.gnfsic = it.GaussNFSIC(gwx2, gwy2, V, W, alpha=0.01)
        self.pdata_mean = pdata_mean

    def test_perform_test(self):
        test_result = self.gnfsic.perform_test(self.pdata_mean)
        # should reject. Cannot assert this for sure.
        #self.assertTrue(test_result['h0_rejected'], 'Test should reject H0')
        pass 

    def test_compute_stat(self):
        stat = self.gnfsic.compute_stat(self.pdata_mean)
        self.assertGreater(stat, 0)


    def tearDown(self):
        pass


class TestQuadHSIC(unittest.TestCase):

    def setUp(self):
        n = 50
        dx = 2
        pdata_mean = get_pdata_mean(n, dx)
        k, l = kl_median(pdata_mean)

        self.qhsic = it.QuadHSIC(k, l, n_permute=60, alpha=0.01)
        self.pdata_mean = pdata_mean

    def test_list_permute(self):
        # test that the permutations are done correctly. 
        # Test against a naive implementation.
        pd = self.pdata_mean
        X, Y = pd.xy()
        k = self.qhsic.k
        l = self.qhsic.l
        n_permute = self.qhsic.n_permute
        s = 113
        arr_hsic = it.QuadHSIC.list_permute(X, Y, k, l, n_permute, seed=s)
        arr_hsic_naive = it.QuadHSIC._list_permute_generic(X, Y, k, l, n_permute, seed=s)
        np.testing.assert_array_almost_equal(arr_hsic, arr_hsic_naive)
                #'Permuted HSIC values are not the same as the naive implementation.')

    def tearDown(self):
        pass

class TestFiniteFeatureHSIC(unittest.TestCase):
    def test_list_permute_spectral(self):
        # make sure that simulating from the spectral approach is roughly the 
        # same as doing permutations.
        ps = data.PS2DSinFreq(freq=2)
        n_features = 5
        n_simulate = 3000
        n_permute = 3000
        for s in [283, 2]:
            with util.NumpySeedContext(seed=s):
                pdata = ps.sample(n=200, seed=s+1)
                X, Y = pdata.xy()

                sigmax2 = 1
                sigmay2 = 0.8
                fmx = fea.RFFKGauss(sigmax2, n_features=n_features, seed=s+3)
                fmy = fea.RFFKGauss(sigmay2, n_features=n_features, seed=s+23)
                ffhsic = it.FiniteFeatureHSIC(fmx, fmy, n_simulate=n_simulate, alpha=0.05, seed=s+89)

                Zx = fmx.gen_features(X)
                Zy = fmy.gen_features(Y)
                list_perm = it.FiniteFeatureHSIC.list_permute(X, Y, fmx, fmy, n_permute=n_permute, seed=s+82)
                list_spectral, eigx, eigy =\
                    it.FiniteFeatureHSIC.list_permute_spectral(Zx, Zy,
                            n_simulate=n_simulate, seed=s+119)

                # make sure that the relative frequency of the histogram does 
                # not differ much.
                freq_p, edge_p = np.histogram(list_perm)
                freq_s, edge_s = np.histogram(list_spectral)
                nfreq_p = freq_p/float(np.sum(freq_p))
                nfreq_s = freq_s/float(np.sum(freq_s))
                arr_diff = np.abs(nfreq_p - nfreq_s)
                self.assertTrue(np.all(arr_diff <= 0.2))

# end class TestFiniteFeatureHSIC


class TestRDC(unittest.TestCase):

    def setUp(self):
        pass 

    def test_rdc(self):
        feature_pairs = 10
        n = 30
        for f in range(1, 7):
            ps = data.PS2DSinFreq(freq=1)
            pdata = ps.sample(n, seed=f+4)
            fmx = fea.RFFKGauss(1, feature_pairs, seed=f+10)
            fmy = fea.RFFKGauss(2.0, feature_pairs+1, seed=f+9)
            rdc = it.RDC(fmx, fmy, alpha=0.01)
            stat, evals = rdc.compute_stat(pdata, return_eigvals=True)

            self.assertGreaterEqual(stat, 0)
            abs_evals = np.abs(evals)
            self.assertTrue(np.all(abs_evals >= 0))
            self.assertTrue(np.all(abs_evals <= 1))


    def tearDown(self):
        pass


class TestFuncs(unittest.TestCase):
    """
    This is to test functions that do not belong to any class. 
    """
    def test_nfsic(self):
        n = 50
        dx = 3
        dy = 1
        X = np.random.randn(n, dx)
        Y = np.random.randn(n, dy) + 1
        medx2 = util.meddistance(X)**2
        medy2 = util.meddistance(Y)**2
        k = kernel.KGauss(medx2)
        l = kernel.KGauss(medy2)
        J = 3
        V = np.random.randn(J, dx)
        W = np.random.randn(J, dy)

        nfsic, mean, cov = it.nfsic(X, Y, k, l, V, W, reg=0)

        self.assertAlmostEqual(np.imag(nfsic), 0)
        self.assertGreater(nfsic, 0)



# end TestQuadHSIC

if __name__ == '__main__':
   unittest.main()
