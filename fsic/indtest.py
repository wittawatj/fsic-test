"""
Module containing many types of independence testing methods.
"""

__author__ = 'wittawat'

from abc import ABCMeta, abstractmethod
from fsic.data import PairedData
import matplotlib.pyplot as plt
import numpy as np
#from numba import jit
import fsic.data as data
import fsic.util as util
import fsic.feature as fea
#from fsic.util import ContextTimer
import fsic.kernel as kernel
import logging
import os

import scipy.stats as stats
import theano
import theano.tensor as tensor
import theano.tensor.nlinalg as nlinalg
import theano.tensor.slinalg as slinalg

class IndTest(object, metaclass=ABCMeta):
    """Abstract class for an independence test for paired sample."""

    def __init__(self, alpha):
        """
        alpha: significance level of the test
        """
        self.alpha = alpha

    @abstractmethod
    def perform_test(self, pdata):
        """perform the two-sample test and return values computed in a dictionary:
        {alpha: 0.01, pvalue: 0.0002, test_stat: 2.3, h0_rejected: True, 
        time_secs: ...}
        pdata: an instance of PairedData
        """
        raise NotImplementedError()

    @abstractmethod
    def compute_stat(self, pdata):
        """Compute the test statistic"""
        raise NotImplementedError()


#------------------------------------------------------

class NFSIC(IndTest):
    """
    Normalized Finite Set Independence Criterion test using the specified kernels 
    and a set of paired test locations.

    H0: X and Y are independent 
    H1: X and Y are dependent.
    """

    def __init__(self, k, l, V, W, alpha=0.01, reg='auto', n_permute=None, seed=87):
        """
        V,W locations are paired.

        :param k: a Kernel on X
        :param l: a Kernel on Y
        :param V: J x dx numpy array of J locations to test the difference
        :param W: J x dy numpy array of J locations to test the difference
        :param n_permute: The number of times to permute the samples to simulate 
            the null distribution. Set to None or 0 to use the asymptotic chi-squared
            distribution. Set to a positive integer to use permutations.
        :param reg: a non-negative regularizer. Can be set to 'auto' to use reg=0 first. 
            If failed, automatically set reg to a low value.
        """
        super(NFSIC, self).__init__(alpha)
        if V.shape[0] != W.shape[0]:
            raise ValueError('number of locations in V and W must be the same.')
        self.V = V 
        self.W = W
        self.k = k
        self.l = l
        self.reg = reg
        self.n_permute = n_permute
        self.seed = seed

    def perform_test(self, pdata):
        if self.n_permute is None or self.n_permute == 0:
            # use asymptotics 
            return self._perform_test_asymptotics(pdata)
        else:
            # assume n_permute > 0
            return self._perform_test_permute(pdata) 


    def _perform_test_permute(self, pdata):
        """
        Perform test by permutations. This is more accurate than using the
        asymptotic null distribution with the sample size is small. However, it
        is slower.
        """
        with util.ContextTimer() as t:
            X, Y = pdata.xy()
            alpha = self.alpha
            nfsic_stat = self.compute_stat(pdata)
            k = self.k
            l = self.l
            n_permute = self.n_permute
            arr_nfsic = NFSIC.list_permute(X, Y, k, l, self.V, self.W, n_permute=n_permute, seed=self.seed, reg=self.reg)
            # approximate p-value with the permutations 
            pvalue = np.mean(arr_nfsic > nfsic_stat)

        results = {'alpha': self.alpha, 'pvalue': pvalue, 'test_stat': nfsic_stat,
                'h0_rejected': pvalue < alpha, 
                #'arr_nfsic': arr_nfsic, 
                'time_secs': t.secs, 'n_permute': n_permute}
        return results

    def _perform_test_asymptotics(self, pdata):
        """
        Perform the test with threshold computed from the chi-squared distribution
        (asymptotic null distribution).
        """
        with util.ContextTimer() as t:
            stat = self.compute_stat(pdata)
            #print('stat: %.3g'%stat)
            J  = self.V.shape[0]
            pvalue = stats.chi2.sf(stat, df=J)
            alpha = self.alpha
        results = {'alpha': self.alpha, 'pvalue': pvalue, 'test_stat': stat,
                'h0_rejected': pvalue < alpha, 'time_secs': t.secs}
        return results

    def compute_stat(self, pdata):
        X, Y = pdata.xy()
        V = self.V
        W = self.W
        k = self.k
        l = self.l
        s, _, _ = nfsic(X, Y, k, l, V, W, reg=self.reg)
        if not np.isfinite(s):
            print('k: %s'%str(k))
            print(('l: %s'%str(l)))
            print(('reg: %s'%str(self.reg)))
            print ('V: ')
            print (V)
            print ('W: ')
            print (W)
            raise ValueError('statistic is not finite. Was %s.'%str(s))

        return s

    @staticmethod 
    def list_permute(X, Y, k, l, V, W, n_permute=400, seed=873, reg=0):
        """
        - reg can be 'auto'. See the description in the constructor.
        """
        if X.shape[0] != Y.shape[0]:
            raise ValueError('X and Y must have the same number of rows (sample size)')
        if V.shape[0] != W.shape[0]:
            raise ValueError('Number of locations in V, W (rows) must be the same.')
        n = X.shape[0]
        J = V.shape[0]

        arr = np.zeros(n_permute)
        K = k.eval(X, V) # n x J
        L = l.eval(Y, W) # n x J
        # mean
        mean_k = np.mean(K, 0)
        mean_l = np.mean(L, 0)
        mkml =  mean_k*mean_l

        rr = reg if np.isreal(reg) else 0
        with util.NumpySeedContext(seed=seed):
            r = 0 
            while r < n_permute:
                # shuffle the order of X, Y while still keeping the original pairs
                ind = np.random.choice(n, n, replace=False)
                Ks = K[ind, :]
                Ls = L[ind, :]
                Kt = Ks - mean_k
                Ktml = Kt*mean_l

                # shift Ls n-1 times 
                for s in range(n-1):
                    if r >= n_permute:
                        break
                    Ls = np.roll(Ls, 1, axis=0)
                    # biased
                    u = np.mean(Ks*Ls, 0) - mkml
                    Snd_mo = Kt*Ls -  Ktml
                    Sig = Snd_mo.T.dot(Snd_mo/n) - np.outer(u, u)
                    st = nfsic_from_u_sig(u, Sig, n, reg)

                    arr[r] = st
                    r = r + 1
        return arr

    @staticmethod
    def _list_permute_naive(X, Y, k, l, V, W, n_permute=400, seed=873, reg=0):
        """
        Simulate from the null distribution by permutations.
        Naive implementation. Just for checking the correctness.
        Return a list of compute statistic for each permutation.
        """
        pdata = data.PairedData(X, Y)
        nfsic = NFSIC(k, l, V, W, reg=reg, n_permute=n_permute, seed=seed)
        arr_nfsic = permute_null_dist(pdata, nfsic, n_permute=n_permute, seed=seed)
        return arr_nfsic


## end class NFSIC

class GaussNFSIC(NFSIC):
    """
    Normalized Finite Set Independence Criterion test using 
    an isotropic Gaussian kernel for each input X and Y.

    H0: X and Y are independent 
    H1: X and Y are dependent.
    """

    def __init__(self, gwidthx, gwidthy, V, W, alpha=0.01, reg='auto', n_permute=None, seed=87):
        """
        V,W locations are paired.

        :param gwidthx: Gaussian width^2 for X
        :param gwidthy: Gaussian width^2 for Y
        :param V: J x dx numpy array of J locations to test the difference
        :param W: J x dy numpy array of J locations to test the difference
        :param n_permute: The number of times to permute the samples to simulate 
            the null distribution. Set to None or 0 to use the asymptotic chi-squared
            distribution. Set to a positive integer to use permutations.
        """
        if V.shape[0] != W.shape[0]:
            raise ValueError('number of locations in V and W must be the same.')
        k = kernel.KGauss(gwidthx)
        l = kernel.KGauss(gwidthy)
        super(GaussNFSIC, self).__init__(k, l, V, W, alpha=alpha, reg=reg,
                n_permute=n_permute, seed=seed)

    @staticmethod
    def func_obj(Xth, Yth, Vth, Wth, gwidthx_th, gwidthy_th, regth, n, J):
        """Return a real-valued
        objective function that works on Theano variables to compute the
        objective to be used for the optimization.    
        - Intended to be used with optimize_locs_widths(..).
        - The returned value is a Theano variable.
        - J is not a Theano variable.

        - regth: regularization parameter
        """
        # shape of a TensorVariable is symbolic until given a concrete value
        diag_regth = regth*tensor.eye(J)
        Kth = GaussNFSIC.gauss_kernel_theano(Xth, Vth, gwidthx_th)
        Lth = GaussNFSIC.gauss_kernel_theano(Yth, Wth, gwidthy_th)
        # mean
        mean_k = Kth.mean(0)
        mean_l = Lth.mean(0)
        KLth = Kth*Lth
        # u is a Theano array
        #from IPython.core.debugger import Tracer 
        #Tracer()()
        #u = (KLth.mean(0) - mean_k*mean_l)*n/(n-1)
        #biased
        u = KLth.mean(0) - mean_k*mean_l

        # cov
        Kt = Kth - mean_k
        Lt = Lth - mean_l 
        # Gam is n x J
        Gam = Kt*Lt - u 
        mean_gam = Gam.mean(0)
        Gam0 = Gam - mean_gam
        Sig = Gam0.T.dot(Gam0)/n
        s = nlinalg.matrix_inverse(Sig + diag_regth).dot(u).dot(u)*n
        #s = nlinalg.matrix_inverse(Sig).dot(u).dot(u)*n
        return s


    @staticmethod
    def gauss_kernel_theano(Xth, Vth, gwidth2):
        """Gaussian kernel. Theano version.
        :return a kernel matrix of size Xth.shape[0] x Vth.shape[0]
        """
        n, d = Xth.shape
        D2 = (Xth**2).sum(1).reshape((-1, 1)) - 2*Xth.dot(Vth.T) +\
            tensor.sum(Vth**2, 1).reshape((1, -1))
        Kth = tensor.exp(-D2/(2.0*gwidth2))
        return Kth


    @staticmethod
    def optimize_locs_widths(pdata, alpha, n_test_locs=5, max_iter=400,
            V_step=1, W_step=1, gwidthx_step=1, gwidthy_step=1,
            batch_proportion=1.0, tol_fun=1e-3, step_pow=0.5, seed=1, reg=1e-5, 
            gwidthx_lb=None, gwidthx_ub=None, gwidthy_lb=None,
        gwidthy_ub=None):
        """Optimize the test locations V, W and the Gaussian kernel width by 
        maximizing a test power criterion. X, Y should not be the same data as
        used in the actual test (i.e., should be a held-out set). 

        - max_iter: #gradient descent iterations
        - batch_proportion: (0,1] value to be multipled with n giving the batch 
            size in stochastic gradient. 1 = full gradient ascent.
        - tol_fun: termination tolerance of the objective value
        - If the lb, ub bounds are None, use fraction of the median heuristics 
            to automatically set the bounds.
        
        Return (V test_locs, W test_locs, gaussian width for x, gaussian width
            for y, info log)
        """

        """
        Optimize the empirical version of Lambda(T) i.e., the criterion used 
        to optimize the test locations, for the test based 
        on difference of mean embeddings with Gaussian kernel. 
        Also optimize the Gaussian width.
        """
        J = n_test_locs
        # Use grid search to initialize the gwidths for both X, Y
        X, Y = pdata.xy()
        n_gwidth_cand = 5
        gwidth_factors = 2.0**np.linspace(-3, 3, n_gwidth_cand) 
        medx2 = util.meddistance(X, 1000)**2
        medy2 = util.meddistance(Y, 1000)**2

        #V, W = GaussNFSIC.init_locs_2randn(pdata, n_test_locs, seed=seed)
        # draw from a joint Gaussian

        # We have to be very careful with init_locs_joint_randn. This freezes 
        # when d > 2000.
        #V, W = GaussNFSIC.init_locs_joint_randn(pdata, n_test_locs, seed=seed)
        #V, W = GaussNFSIC.init_locs_joint_subset(pdata, n_test_locs, seed=seed)

        k = kernel.KGauss(medx2*2)
        l = kernel.KGauss(medy2*2)
        V, W = GaussNFSIC.init_check_subset(pdata, J, k, l, n_cand=20, subsample=2000,
           seed=seed+27)
        #V, W = GaussNFSIC.init_locs_marginals_subset(pdata, n_test_locs, seed=seed)

        list_gwidthx = np.hstack( ( (medx2)*gwidth_factors ) )
        list_gwidthy = np.hstack( ( (medy2)*gwidth_factors ) )
        bestij, lambs = GaussNFSIC.grid_search_gwidth(pdata, V, W,
                list_gwidthx, list_gwidthy)
        gwidthx0 = list_gwidthx[bestij[0]]
        gwidthy0 = list_gwidthy[bestij[1]]
        assert util.is_real_num(gwidthx0), 'gwidthx0 not real. Was %s'%str(gwidthx0)
        assert util.is_real_num(gwidthy0), 'gwidthy0 not real. Was %s'%str(gwidthy0)
        assert gwidthx0 > 0, 'gwidthx0 not positive. Was %.3g'%gwidthx0
        assert gwidthy0 > 0, 'gwidthy0 not positive. Was %.3g'%gwidthy0
        logging.info('After grid search, gwidthx0=%.3g'%gwidthx0)
        logging.info('After grid search, gwidthy0=%.3g'%gwidthy0)

        # set the width bounds
        fac_min = 5e-2
        fac_max = 5e3
        gwidthx_lb = gwidthx_lb if gwidthx_lb is not None else fac_min*medx2
        gwidthx_ub = gwidthx_ub if gwidthx_ub is not None else fac_max*medx2
        gwidthy_lb = gwidthy_lb if gwidthy_lb is not None else fac_min*medy2
        gwidthy_ub = gwidthy_ub if gwidthy_ub is not None else fac_max*medy2

        # info = optimization info 
        V, W, gwidthx, gwidthy, info  = \
                generic_optimize_locs_widths(pdata, V, W, gwidthx0, gwidthy0,
                        GaussNFSIC.func_obj, max_iter=max_iter, V_step=V_step,
                        W_step=W_step, gwidthx_step=gwidthx_step,
                        gwidthy_step=gwidthy_step,
                        batch_proportion=batch_proportion, tol_fun=tol_fun,
                        step_pow=step_pow, reg=reg, seed=seed+1, 
                        gwidthx_lb=gwidthx_lb, gwidthx_ub=gwidthx_ub,
                        gwidthy_lb=gwidthy_lb, gwidthy_ub=gwidthy_ub)
        assert util.is_real_num(gwidthx), 'gwidthx is not real. Was %s' % str(gwidthx)
        assert util.is_real_num(gwidthy), 'gwidthy is not real. Was %s' % str(gwidthy)

        # make sure that the optimized gwidthx, gwidthy are not too small
        # or too large.
        fac_min = 5e-2 
        fac_max = 5e3
        gwidthx = max(fac_min*medx2, 1e-7, min(fac_max*medx2, gwidthx))
        gwidthy = max(fac_min*medy2, 1e-7, min(fac_max*medy2, gwidthy))
        return V, W, gwidthx, gwidthy, info

    @staticmethod
    def grid_search_gwidth(pdata, V, W, list_gwidthx, list_gwidthy):
        """
        Linear search for the best Gaussian width in the list that maximizes 
        the test power criterion, fixing the test locations. 

        - V: a J x dx np-array for J test locations for X.
        - W: a J x dy np-array for J test locations for Y.

        return: (best width index, list of test power objectives)
        """
        list_gauss_kernelx = [kernel.KGauss(gw) for gw in list_gwidthx]
        list_gauss_kernely = [kernel.KGauss(gw) for gw in list_gwidthy]
        besti, objs = nfsic_grid_search_kernel(pdata, V, W, list_gauss_kernelx,
                list_gauss_kernely)
        return besti, objs

    @staticmethod
    def init_locs_marginals_subset(pdata, n_test_locs, seed=3):
        """
        Choose n_test_locs points randomly from each of the two marginal samples.
        Thus, pairs are not maintained.
        """
        n = pdata.sample_size()
        Ix = util.subsample_ind(n, n_test_locs, seed=seed)
        Iy = util.subsample_ind(n, n_test_locs, seed=seed+28)
        X, Y = pdata.xy()
        V = X[Ix, :]
        W = Y[Iy, :]
        return V, W

    @staticmethod
    def init_check_subset(pdata, n_test_locs, k, l, n_cand=50, subsample=2000,
            seed=3):
        """
        Evaluate a set of locations to find the best locations to initialize. 
        The location candidates are drawn from the joint and the product of 
        the marginals.
        - subsample the data when computing the objective 
        - n_cand: number of times to draw from the joint and the product 
            of the marginals.
        Return V, W
        """

        X, Y = pdata.xy()
        n = pdata.sample_size()

        # from the joint 
        objs_joint = np.zeros(n_cand)
        seed_seq_joint = util.subsample_ind(5*n_cand, n_cand, seed=seed*5)
        for i in range(n_cand):
            V, W = GaussNFSIC.init_locs_joint_subset(pdata, n_test_locs, seed=seed_seq_joint[i])
            if subsample < n:
                I = util.subsample_ind(n, n_test_locs, seed=seed_seq_joint[i]+1)
                XI = X[I, :]
                YI = Y[I, :]
            else:
                XI = X
                YI = Y
            objs_joint[i], _, _ = nfsic(XI, YI, k, l, V, W, reg='auto')
        objs_joint[np.logical_not(np.isfinite(objs_joint))] = -np.infty
        logging.info(objs_joint)
        # best index 
        bind_joint = np.argmax(objs_joint)

        # from the product of the marginals 
        objs_prod = np.zeros(n_cand)
        seed_seq_prod = util.subsample_ind(5*n_cand, n_cand, seed=seed*3+1)
        for i in range(n_cand):
            V, W = GaussNFSIC.init_locs_marginals_subset(pdata, n_test_locs, seed=seed_seq_prod[i])
            if subsample < n:
                I = util.subsample_ind(n, n_test_locs, seed=seed_seq_prod[i]+1)
                XI = X[I, :]
                YI = Y[I, :]
            else:
                XI = X
                YI = Y
            objs_prod[i], _, _ = nfsic(XI, YI, k, l, V, W, reg='auto')

        objs_prod[np.logical_not(np.isfinite(objs_prod))] = -np.infty
        logging.info(objs_prod)
        bind_prod = np.argmax(objs_prod)

        if objs_joint[bind_joint] >= objs_prod[bind_prod]:
            V, W = GaussNFSIC.init_locs_joint_subset(pdata, n_test_locs,
                    seed=seed_seq_joint[bind_joint])
        else:
            V, W = GaussNFSIC.init_locs_marginals_subset(pdata, n_test_locs,
                    seed=seed_seq_prod[bind_prod])
        return V, W



    @staticmethod
    def init_locs_joint_subset(pdata, n_test_locs, seed=2):
        """
        Choose n_test_locs points randomly from the joint sample.
        Pairs are maintained.

        There are a few advantages of this approach over fitting a Gaussian 
        and drawing from it.
        - Faster and simpler 
        - Guarantee that V, W will not leave the boundary of the sample.
            This can make NFSIC covariance singular.
        """
        n = pdata.sample_size()
        I = util.subsample_ind(n, n_test_locs, seed=seed)
        X, Y = pdata.xy()
        V = X[I, :]
        W = Y[I, :]
        return V, W

    @staticmethod 
    def init_locs_joint_randn(pdata, n_test_locs, subsample=2000, seed=1):
        """
        Fit a joint Gaussian to (X, Y) and draw n_test_locs.
        return (V, W) each containing n_test_locs vectors. 
        """
        #import pdb; pdb.set_trace()

        X, Y = pdata.xy()
        n = pdata.sample_size()
        sub = min(n, subsample)
        if sub < n:
            X = util.subsample_rows(X, sub, seed=seed+1)
            Y = util.subsample_rows(Y, sub, seed=seed+2)

        XY = np.hstack((X, Y))
        dx = X.shape[1]
        dy = Y.shape[1]
        VW = util.fit_gaussian_draw(XY, n_test_locs, seed=seed+8, eig_pow=0.9)
        V = VW[:, :dx]
        W = VW[:, dx:]

        # make sure v, W live within the boundary of the data 
        V = util.bound_by_data(V, X)
        W = util.bound_by_data(W, Y)
        return V, W

    @staticmethod 
    def init_locs_2randn(pdata, n_test_locs, subsample=2000, seed=1):
        """
        Fit a Gaussian to each dataset of X and Y, and draw 
        n_test_locs from each. 

        return (V, W) each containing n_test_locs vectors drawn from their
        respective Gaussian fit.
        """

        X, Y = pdata.xy()
        n = pdata.sample_size()
        sub = min(n, subsample)
        if sub < n:
            X = util.subsample_rows(X, sub, seed=seed+1)
            Y = util.subsample_rows(Y, sub, seed=seed+2)
        V = util.fit_gaussian_draw(X, n_test_locs, seed=seed)
        W = util.fit_gaussian_draw(Y, n_test_locs, seed=seed+29)

        # make sure v, W live within the boundary of the data 
        V = util.bound_by_data(V, X)
        W = util.bound_by_data(W, Y)

        #from IPython.core.debugger import Tracer
        #t = Tracer()
        #t()
        return V, W

# end of class GaussNFSIC


def generic_optimize_locs_widths(pdata, V0, W0, gwidthx0, gwidthy0,
        func_obj, max_iter=400, V_step=1.0, W_step=1.0, gwidthx_step=1.0,
        gwidthy_step=1.0, batch_proportion=1.0, tol_fun=1e-3, step_pow=0.5,
        reg=1e-5, seed=101, gwidthx_lb=1e-3, gwidthx_ub=1e6, gwidthy_lb=1e-3,
        gwidthy_ub=1e6):
    """Optimize the test locations V, W and the Gaussian kernel width by 
    maximizing a test power criterion. X, Y should not be the same data as used 
    in the actual test (i.e., should be a held-out set). 
    Optimize the empirical version of the test statistic Lambda(T).

    - V0, W0: Jxdx and Jxdy numpy arrays. Initial values of V, W test locations.
      J = the number of test locations/frequencies
    - gwidthx0, gwidthy0: initial Gaussian widths 
    - func_obj: (X, Y, V, W, gwidthx, gwidthy, reg, J) |-> a real-valued objective
    function that works on Theano variables to compute the objective to be used
    for the optimization.    
    - max_iter: #gradient descent iterations
    - batch_proportion: (0,1] value to be multipled with n giving the batch 
        size in stochastic gradient. 1 = full gradient ascent.
    - tol_fun: termination tolerance of the objective value
    - step_pow: in var <- var + step_size*gradient/iteration**step_pow
    - gwidthx_lb: lower bound for gwidthx to maintain at all time during the
      gradient ascent.
    
    Return optimized (V, W, gwidthx, gwidthy, info log)
    """
    # Running theano with multiple processes trying to access the same 
    # function can create a problem.  This is an attempt to solve it.
    #theano.gof.compilelock.set_lock_status(False)

    def gwidth_constrain(var, lb, ub):
        """
        Make sure that the Theano variable var is inside the interval defined 
        by lower bound lb, and upper bound ub. Perform a project if it does
        not.
        """
        var = tensor.minimum(var, ub)
        var = tensor.maximum(var, lb)
        return var

    if V0.shape[0] != W0.shape[0]:
        raise ValueError('V0 and W0 must have the same number of rows J.')

    X, Y = pdata.xy()
    # initialize Theano variables
    Vth = theano.shared(V0, name='V')
    Wth = theano.shared(W0, name='W')

    Xth = tensor.dmatrix('X')
    Yth = tensor.dmatrix('Y')
    it = theano.shared(1, name='iter')
    # square root of the Gaussian width. Use square root to handle the 
    # positivity constraint by squaring it later.
    gwidthx_sq0 = gwidthx0**0.5
    gwidthx_sq_th = theano.shared(gwidthx_sq0, name='gwidthx_sq')
    gwidthy_sq0 = gwidthy0**0.5
    gwidthy_sq_th = theano.shared(gwidthy_sq0, name='gwidthy_sq')
    regth = theano.shared(reg, name='reg')

    #sqr(x) = x^2
    s = func_obj(Xth, Yth, Vth, Wth, tensor.sqr(gwidthx_sq_th),
            tensor.sqr(gwidthy_sq_th), regth, X.shape[0], V0.shape[0])

    g_V, g_W, g_gwidthx_sq, g_gwidthy_sq = tensor.grad(s, [Vth, Wth,
        gwidthx_sq_th, gwidthy_sq_th])

    # heuristic to prevent the step sizes from being too large
    max_gwidthx_step = np.amin(np.std(X, 0))/2.0
    max_gwidthy_step = np.amin(np.std(Y, 0))/2.0
    functh = theano.function(inputs=[Xth, Yth], outputs=s, 
           updates=[
              (Vth, Vth+V_step*g_V/it**step_pow/tensor.sum(g_V**2)**0.5 ), 
              (Wth, Wth+W_step*g_W/it**step_pow/tensor.sum(g_W**2)**0.5 ), 
              (it, it+1), 
              #(gamma_sq_th, gamma_sq_th+gwidthx_step*gra_gamma_sq\
              #        /it**step_pow/tensor.sum(gra_gamma_sq**2)**0.5 ) 
              (gwidthx_sq_th, 
                  gwidth_constrain(
                      gwidthx_sq_th+gwidthx_step*tensor.sgn(g_gwidthx_sq) \
                      *tensor.minimum(tensor.abs_(g_gwidthx_sq), max_gwidthx_step) \
                      /it**step_pow,
                      tensor.sqrt(gwidthx_lb), tensor.sqrt(gwidthx_ub)
                  )
              ), 
              (gwidthy_sq_th, 
                  gwidth_constrain(
                      gwidthy_sq_th+gwidthy_step*tensor.sgn(g_gwidthy_sq) \
                     *tensor.minimum(tensor.abs_(g_gwidthy_sq), max_gwidthy_step) \
                     /it**step_pow, 
                     tensor.sqrt(gwidthy_lb), tensor.sqrt(gwidthy_ub)
                 )
              ) 
              ] 
           )
    # copy the function with the hope that multiple processes can access their own 
    # compiled functions. Did not work?
    #http://deeplearning.net/software/theano/library/compile/function.html#theano.compile.function_module.Function.copy
    #pid = os.getpid()
    #fullhost = '_'.join(os.uname())
    #functh = functh.copy(share_memory=False, name='fsic_func_%s_%s'%(fullhost, str(pid)))

    # //////// run gradient ascent //////////////
    # S for statistics
    S = np.zeros(max_iter)
    J = V0.shape[0]
    _, dx = X.shape
    n, dy = Y.shape
    Vs = np.zeros((max_iter, J, dx))
    Ws = np.zeros((max_iter, J, dy))
    gwidthxs = np.zeros(max_iter)
    gwidthys = np.zeros(max_iter)

    logging.info('Iterating gradient ascent')
    with util.NumpySeedContext(seed=seed):
        for t in range(max_iter):
            # stochastic gradient ascent (or full gradient ascent)
            ind = np.random.choice(n, min(int(batch_proportion*n), n), replace=False)
            # record objective values 
            try:
                S[t] = functh(X[ind, :], Y[ind, :])
            except: 
                print('Exception occurred during gradient descent. Stop optimization.')
                print('Return the value from previous iter. ')
                import traceback as tb 
                tb.print_exc()
                t = t -1
                break

            Vs[t] = Vth.get_value()
            Ws[t] = Wth.get_value()
            gwidthxs[t] = gwidthx_sq_th.get_value()**2
            gwidthys[t] = gwidthy_sq_th.get_value()**2

            logging.info('t: %d. obj: %.4g, gwx: %.4g, gwy: %.4g', t, S[t],
                gwidthxs[t], gwidthys[t])
            #logging.info('V')
            #logging.info(Vs[t])
            #logging.info('W')
            #logging.info(Ws[t])

            # check the change of the objective values 
            if t >= 4 and abs(S[t]-S[t-1]) <= tol_fun:
                break

    S = S[:t+1]
    Vs = Vs[:t+1]
    Ws = Ws[:t+1]
    gwidthxs = gwidthxs[:t+1]
    gwidthys = gwidthys[:t+1]

    # optimization info 
    info = {'Vs': Vs, 'Ws': Ws, 'V0':V0, 'W0': W0, 'gwidthxs': gwidthxs,
            'gwidthys': gwidthys,  'gwidthx0': gwidthx0, 'gwidthy0': gwidthy0,
            'obj_values': S}

    if t >= 0:
        opt_V = Vs[-1]
        opt_W = Ws[-1]
        # for some reason, optimization can give a non-numerical result
        opt_gwidthx = gwidthxs[-1] if util.is_real_num(gwidthxs[-1]) else gwidthx0
        opt_gwidthy = gwidthys[-1] if util.is_real_num(gwidthys[-1]) else gwidthy0

    else:
        # Probably an error occurred in the first iter.
        logging.warning('t=%d. gwx0=%.3g, gwy0=%.3g'%(t, gwidthx0, gwidthy0))
        
        opt_V = V0
        opt_W = W0
        opt_gwidthx = gwidthx0
        opt_gwidthy = gwidthy0
    return (opt_V, opt_W, opt_gwidthx, opt_gwidthy, info  )


def nfsic_grid_search_kernel(pdata, V, W, list_kernelx, list_kernely):
    """
    Linear search for the best Gaussian width in the list that maximizes 
    the test power criterion, fixing the test locations to (V, W)

    - list_kernelx: list of kernel candidates for X 
    - list_kernely: list of kernel candidates for Y

    Perform a mesh grid on the two lists.

    TODO: Can be made faster by computing variables depending on k only once 
        for each k candidate.

    return: (best kernel index pair (i,j), 2d-array of test power criteria of size 
      len(list_kernelx) x len(list_kernely) )
    """
    # number of test locations
    if V.shape[0] != W.shape[0]:
        raise ValueError('V and W must have the same number of rows.')

    X, Y = pdata.xy()
    n = X.shape[0]
    J = V.shape[0]
    n_cand_x = len(list_kernelx)
    n_cand_y = len(list_kernely)
    lambs = np.zeros((n_cand_x, n_cand_y))
    for i in range(n_cand_x):
        k = list_kernelx[i]
        K = k.eval(X, V) # n x J
        mean_k = np.mean(K, 0)
        Kt = K - mean_k

        for j in range(n_cand_y):
            l = list_kernely[j]
            L = l.eval(Y, W) # n x J
            try:
                # mean
                mean_l = np.mean(L, 0)

                # biased
                u = np.mean(K*L, 0) - mean_k*mean_l
                # cov
                Lt = L - mean_l 

                Snd_mo = Kt*Lt 
                Sig = Snd_mo.T.dot(Snd_mo)/n - np.outer(u, u)

                lamb = nfsic_from_u_sig(u, Sig, n, reg='auto')
                if lamb <= 0:
                    # This can happen when Z, Sig are ill-conditioned. 
                    #print('negative lamb: %.3g'%lamb)
                    raise np.linalg.LinAlgError
                #from IPython.core.debugger import Tracer 
                #Tracer()()
                if np.iscomplex(lamb):
                    # complex value can happen if the covariance is ill-conditioned?
                    print(('Lambda is complex. Truncate the imag part. lamb: %s'%(str(lamb))))
                    lamb = np.real(lamb)

                lambs[i, j] = lamb
                logging.info('(%d, %d), lamb: %5.4g, kx: %s, ky: %s ' %(i, j, lamb,
                   str(k), str(l) ))
            except np.linalg.LinAlgError:
                # probably matrix inverse failed. 
                lambs[i, j] = np.NINF

    #Widths that come early in the list 
    # are preferred if test powers are equal.
    bestij = np.unravel_index(lambs.argmax(), lambs.shape)
    return bestij, lambs

def nfsic_from_u_sig(u, Sig, n, reg=0):
    """
    Compute the NFSIC statistic from the u vector, and the covariance matrix 
    Sig. reg can be 'auto'. See nfsic().
    Return the statistic.
    """
    J = len(u)
    if J==1:
        r = reg if np.isreal(reg) else 0
        s = float(n)*(u[0]**2)/(r + Sig[0,0])
    else:
        if reg=='auto':
            # First compute with reg=0. If no problem, do nothing. 
            # If the covariance is singular, make 0 eigenvalues positive.
            try:
                s = n*np.linalg.solve(Sig, u).dot(u)        
            except np.linalg.LinAlgError:
                try:
                    # singular matrix 
                    # eigen decompose
                    evals, eV = np.linalg.eig(Sig)
                    evals = np.real(evals)
                    eV = np.real(eV)
                    evals = np.maximum(0, evals)
                    # find the non-zero second smallest eigenvalue
                    snd_small = np.sort(evals[evals > 0])[0]
                    evals[evals <= 0] = snd_small

                    # reconstruct Sig 
                    Sig = eV.dot(np.diag(evals)).dot(eV.T)
                    # try again
                    s = n*np.linalg.solve(Sig, u).dot(u)        
                except:
                    s = np.nan
        else:
            # assume reg is a number 
            #evals, _ = np.linalg.eig(Sig)
            #print np.min(evals)
            s = n*np.linalg.solve(Sig + reg*np.eye(Sig.shape[0]), u).dot(u) 
    return s


def nfsic(X, Y, k, l, V, W, reg=0):
    """
    X: n x dx data matrix 
    Y: n x dy data matrix
    k: kernel for X
    l : kernel for Y
    V: J x dx test locations for X
    W: J x dy test locations for Y 
    reg: a non-negative regularizer. Can be set to 'auto' to use reg=0 first. 
        If failed, automatically set reg to a low value.
    :return (stat, mean, cov)
    """
    assert X.shape[0] == Y.shape[0]
    assert V.shape[0] == W.shape[0]
    n = X.shape[0]
    J = V.shape[0]

    K = k.eval(X, V) # n x J
    L = l.eval(Y, W) # n x J
    # mean
    mean_k = np.mean(K, 0)
    mean_l = np.mean(L, 0)
    #u = float(n)/(n-1)*(np.mean(K*L, 0) - mean_k*mean_l)

    # biased
    u = np.mean(K*L, 0) - mean_k*mean_l
    # cov
    #if use_h1_cov:
    # Assume H_1 i.e., not assuming that the joint factorizes.
    # Generic covariance.
    Kt = K - mean_k
    Lt = L - mean_l 

    Snd_mo = Kt*Lt 
    Sig = Snd_mo.T.dot(Snd_mo)/n - np.outer(u, u)
    s = nfsic_from_u_sig(u, Sig, n, reg)
    return s, u, Sig


class QuadHSIC(IndTest):
    """
    An independece test with Hilbert Schmidt Independence Criterion (HSIC) using
    permutations. This is the test originally proposed in Gretton et al.,
    2005 "Measuring Statistical Dependence with Hilbert-Schmidt Norms"
    Use the biased estimator for HSIC.

    H0: X and Y are independent 
    H1: X and Y are dependent.
    """

    def __init__(self, k, l, n_permute=400, alpha=0.01, seed=87):
        """
        :param k: a Kernel on X
        :param l: a Kernel on Y
        """
        super(QuadHSIC, self).__init__(alpha)
        self.k = k
        self.l = l
        self.n_permute = n_permute
        self.seed = seed

    def perform_test(self, pdata):
        with util.ContextTimer() as t:
            alpha = self.alpha
            bhsic_stat = self.compute_stat(pdata)

            X, Y = pdata.xy()
            k = self.k
            l = self.l
            n_permute = self.n_permute
            arr_hsic = QuadHSIC.list_permute(X, Y, k, l, n_permute, seed=self.seed)
            # approximate p-value with the permutations 
            pvalue = np.mean(arr_hsic > bhsic_stat)

        results = {'alpha': self.alpha, 'pvalue': pvalue, 'test_stat': bhsic_stat,
                'h0_rejected': pvalue < alpha, 
                #'arr_bhsic': arr_hsic, 
                'time_secs': t.secs, 'n_permute': n_permute}
        return results

    def compute_stat(self, pdata):
        X, Y = pdata.xy()
        k = self.k
        l = self.l
        bhsic = QuadHSIC.biased_hsic(X, Y, k, l)
        return bhsic

    @staticmethod
    def biased_hsic(X, Y, k, l): 
        """
        Compute the biased estimator of HSIC as in Gretton et al., 2005.

        :param k: a Kernel on X
        :param l: a Kernel on Y
        """
        if X.shape[0] != Y.shape[0]:
            raise ValueError('X and Y must have the same number of rows (sample size')

        n = X.shape[0]
        K = k.eval(X, X)
        L = l.eval(Y, Y)
        Kmean = np.mean(K, 0)
        Lmean = np.mean(L, 0)
        HK = K - Kmean
        HL = L - Lmean
        # t = trace(KHLH)
        HKf = HK.flatten()/(n-1) 
        HLf = HL.T.flatten()/(n-1)
        hsic = HKf.dot(HLf)
        #t = HK.flatten().dot(HL.T.flatten())
        #hsic = t/(n-1)**2.0
        return hsic


    @staticmethod 
    def list_permute(X, Y, k, l, n_permute=400, seed=8273):
        """
        Repeatedly mix, permute X, Y so that pairs are broken, and compute HSIC.
        This is intended to be used to approximate the null distribution.

        Return a numpy array of HSIC's for each permutation.
        """
        #return QuadHSIC._list_permute_generic(X, Y, k, l, n_permute, seed)
        return QuadHSIC._list_permute_preKL(X, Y, k, l, n_permute, seed)

    @staticmethod
    def _list_permute_preKL(X, Y, k, l, n_permute=400, seed=8273):
        """
        Return a numpy array of HSIC's for each permutation.

        This is an implementation where kernel matrices are pre-computed.

        TODO: can be improved.
        """
        if X.shape[0] != Y.shape[0]:
            raise ValueError('X and Y must have the same number of rows (sample size')
        n = X.shape[0]

        r = 0 
        arr_hsic = np.zeros(n_permute)
        K = k.eval(X, X)
        L = l.eval(Y, Y)
        # set the seed 
        rand_state = np.random.get_state()
        np.random.seed(seed)

        while r < n_permute:
            # shuffle the order of X, Y while still keeping the original pairs
            ind = np.random.choice(n, n, replace=False)
            Ks = K[np.ix_(ind, ind)]
            #Xs = X[ind]
            #Ys = Y[ind]
            #Ks2 = k.eval(Xs, Xs)
            #assert np.linalg.norm(Ks - Ks2, 'fro') < 1e-4

            Ls = L[np.ix_(ind, ind)]
            Kmean = np.mean(Ks, 0)
            HK = Ks - Kmean
            HKf = HK.flatten()/(n-1) 
            # shift Ys n-1 times 
            for s in range(n-1):
                if r >= n_permute:
                    break
                Ls = np.roll(Ls, 1, axis=0)
                Ls = np.roll(Ls, 1, axis=1)

                # compute HSIC 
                Lmean = np.mean(Ls, 0)
                HL = Ls - Lmean
                # t = trace(KHLH)
                HLf = HL.T.flatten()/(n-1)
                bhsic = HKf.dot(HLf)

                arr_hsic[r] = bhsic
                r = r + 1
        # reset the seed back 
        np.random.set_state(rand_state)
        return arr_hsic


    @staticmethod 
    def _list_permute_generic(X, Y, k, l, n_permute=400, seed=8273):
        """
        Return a numpy array of HSIC's for each permutation.

        This is a naive generic implementation where kernel matrices are 
        not pre-computed.
        """
        if X.shape[0] != Y.shape[0]:
            raise ValueError('X and Y must have the same number of rows (sample size')
        n = X.shape[0]

        r = 0 
        arr_hsic = np.zeros(n_permute)
        # set the seed 
        rand_state = np.random.get_state()
        np.random.seed(seed)
        while r < n_permute:
            # shuffle the order of X, Y while still keeping the original pairs
            ind = np.random.choice(n, n, replace=False)
            Xs = X[ind]
            Ys = Y[ind]
            # shift Ys n-1 times 
            for s in range(n-1):
                if r >= n_permute:
                    break
                Ys = np.roll(Ys, 1, axis=0)
                # compute HSIC 
                bhsic = QuadHSIC.biased_hsic(Xs, Ys, k, l)
                arr_hsic[r] = bhsic
                r = r + 1
        # reset the seed back 
        np.random.set_state(rand_state)
        return arr_hsic

# end class QuadHSIC


class FiniteFeatureHSIC(IndTest):
    """
    An independence test with Hilbert Schmidt Independence Criterion (HSIC) using 
    finite dimensional kernels. Explicit feature maps are used to generate 
    features. 
    - The statistic is n*HSIC_biased^2

    Reference: 
    Large-Scale Kernel Methods for Independence Testing
    Qinyi Zhang, Sarah Filippi,  Arthur Gretton, Dino Sejdinovic

    H0: X and Y are independent 
    H1: X and Y are dependent.
    """
    def __init__(self, fmx, fmy, alpha=0.01, n_simulate=5000, seed=90):
        """
        fmx: a FeatureMap for X 
        fmy: a FeatureMap for Y 
        """
        super(FiniteFeatureHSIC, self).__init__(alpha)
        self.fmx = fmx 
        self.fmy = fmy 
        self.n_simulate = n_simulate
        self.seed = seed

    def perform_test(self, pdata):
        with util.ContextTimer() as t:
            alpha = self.alpha
            ffhsic = self.compute_stat(pdata)
            X, Y = pdata.xy()
            Zx = self.fmx.gen_features(X)
            Zy = self.fmy.gen_features(Y)

            n_simulate = self.n_simulate
            arr_nhsic, eigx, eigy = FiniteFeatureHSIC.list_permute_spectral(Zx, Zy, n_simulate, seed=self.seed)
            #arr_nhsic = FiniteFeatureHSIC.list_permute(X, Y, self.fmx, self.fmy, n_permute=n_simulate, seed=self.seed)
            # approximate p-value with the permutations 
            pvalue = np.mean(arr_nhsic > ffhsic)

        results = {'alpha': self.alpha, 'pvalue': pvalue, 'test_stat': ffhsic,
                'h0_rejected': pvalue < alpha, 
                #'arr_nhsic': arr_nhsic, 
                'time_secs': t.secs, 'n_simulate': n_simulate}
        return results

    def compute_stat(self, pdata):
        # complexity = O(Dx*Dy*n)
        # Dx = number of features of X
        X, Y = pdata.xy()
        n = pdata.sample_size()
        Zx = self.fmx.gen_features(X)
        Zy = self.fmy.gen_features(Y)
        HZy = Zy - np.mean(Zy, 0)
        #HZx = Zx - np.mean(Zx, 0)
        M = Zx.T.dot(HZy)/n
        stat = np.sum(M**2)*n
        if np.abs(np.imag(stat)) < 1e-8:
            stat = stat.real
        else:
            raise ValueError('Test statistic is imaginary. Why?')
        return stat

    @staticmethod 
    def list_permute_spectral(features_x, features_y, n_simulate=5000, seed=8274):
        """
        Simulate the null distribution using the spectrums of the cross covariance 
        operator. See theorems in the reference.
        This is intended to be used to approximate the null distribution.

        - features_x: n x Dx where Dx is the number of features for x 
        - features_y: n x Dy

        Return (a numpy array of simulated HSIC values, eigenvalues of X, eigenvalues of Y)
        """
        if features_x.shape[0] != features_y.shape[0]:
            raise ValueError('features_x, features_y must have the same number of rows n.')
        n = features_x.shape[0]
        Dx = features_x.shape[1]
        Dy = features_y.shape[1]
        # The spectrum of the cross covariance operator is given by the product
        # of the spectrums of H*K_x*H and H*_K_y*H.
        HZx = features_x - np.mean(features_x, 0) # n x Dx
        HZy = features_y - np.mean(features_y, 0)
        # spectrum of H*K_x*H \approx spectrum of Zx.T*H*Z_x
        DxDx = HZx.T.dot(HZx)
        DyDy = HZy.T.dot(HZy)

        # eigen decompose 
        eigx, _ = np.linalg.eig(DxDx)
        eigx = np.real(eigx)
        eigy, _ = np.linalg.eig(DyDy)
        eigy = np.real(eigy)
        # sort in decreasing order 
        eigx = -np.sort(-eigx)
        eigy = -np.sort(-eigy)
        # Estimated eigenvalues of the true covariance operators have to be rescaled.
        # See "A Fast, Consistent Kernel Two-Sample Test." Gretton et al., 2009
        # Eq. 5
        eigx = eigx/n
        eigy = eigy/n

        sim_hsics = FiniteFeatureHSIC.simulate_null_dist(eigx, eigy,
                biased_hsic=True, n_simulate=n_simulate, seed=seed)
        return sim_hsics, eigx, eigy

    @staticmethod 
    def simulate_null_dist(eigx, eigy, biased_hsic=True, n_simulate=5000, seed=7):
        """
        Simulate the null distribution using the spectrums of the cross covariance 
        operator. These are determined by the spectrum of the covariance operator 
        of X (eigx), and of Y (eigy). The simulated statistic is
        n*HSIC^2 where HSIC can be biased or unbiased (depending on
        the input argument).

        Reference: Theorem 1 of 
            Large-Scale Kernel Methods for Independence Testing
            Qinyi Zhang, Sarah Filippi,  Arthur Gretton, Dino Sejdinovic

        - eigx: a numpy array of estimated eigenvalues of the covariance operator of X. 
            These are the eigenvalues of H*K*H divided by n.
        - eigy: a numpy array of eigenvalues of the covariance operator of Y

        Return a numpy array of simulated statistics.
        """
        # draw at most Dx,Dy x block_size values at a time
        block_size = 400
        Dx = len(eigx)
        Dy = len(eigy)
        hsics = np.zeros(n_simulate)
        from_ind = 0
        bias_shift = np.sum(eigx)*np.sum(eigy)
        ex_ey = np.outer(eigx, eigy).reshape(-1)
        with util.NumpySeedContext(seed=seed):
            while from_ind < n_simulate:
                to_draw = min(block_size, n_simulate-from_ind)
                # draw chi^2 random variables. 
                chi2 = np.random.randn(Dx*Dy, to_draw)**2
                # an array of length to_draw 
                sim_hsics = ex_ey.dot(chi2)
                if not biased_hsic:
                    # for unbiased statistics
                    sim_hsics = sim_hsics - bias_shift
                # store 
                end_ind = from_ind+to_draw
                hsics[from_ind:end_ind] = sim_hsics
                from_ind = end_ind
        return hsics


    @staticmethod 
    def list_permute(X, Y, fmx, fmy, n_permute=400, seed=872):
        """
        Simulate from the null distribution by permutations.
        Naive implementation. Just for checking the correctness.
        """
        pdata = data.PairedData(X, Y)
        fhsic = FiniteFeatureHSIC(fmx, fmy)
        n = pdata.sample_size()
        null_ps = data.PSNullResample(pdata)
        arr_hsic = np.zeros(n_permute)

        sub_seeds = util.subsample_ind(n_permute*2, n_permute, seed=seed)
        for i in range(n_permute):
            null_pdata = null_ps.sample(n, sub_seeds[i])
            stat = fhsic.compute_stat(null_pdata)
            arr_hsic[i] = stat 
        return arr_hsic

# end class FiniteFeatureHSIC


class NystromHSIC(FiniteFeatureHSIC):
    """
    An independence test with Hilbert Schmidt Independence Criterion (HSIC) using 
    Nystrom approximation.
    features. 
    - The statistic is n*HSIC_biased^2

    Reference: 
    Large-Scale Kernel Methods for Independence Testing
    Qinyi Zhang, Sarah Filippi,  Arthur Gretton, Dino Sejdinovic

    H0: X and Y are independent 
    H1: X and Y are dependent.
    """
    def __init__(self, k, l, induce_x, induce_y, n_simulate=5000, alpha=0.01,
            seed=92):
        """
        k: kernel for X
        l: kernel for Y
        induce_x: Dx x dx inducing points for X. Dx = #inducing points.
        induce_y: Dy x dy incuding points for Y.
        """
        fmx = fea.NystromFeatureMap(k, induce_x)
        fmy = fea.NystromFeatureMap(l, induce_y)
        super(NystromHSIC, self).__init__(fmx, fmy, alpha=alpha,
                n_simulate=n_simulate, seed=seed)
        self.k = k 
        self.l = l


class RDC(IndTest):
    """
    An independece test with the Randomized Dependence Coefficient (Lopez-Paz et
    al., 2013). Use Bartlett's approximation to approximate the null distribution. 
    No permutation.
    - The Bartlett's approximation does not seem to be accurate. See RDCPerm
      instead (use permutations to simulate from the null distribution).

    H0: X and Y are independent 
    H1: X and Y are dependent.
    """

    def __init__(self, fmx, fmy, alpha=0.01):
        """
        fmx: a FeatureMap for X 
        fmy: a FeatureMap for Y 
        """
        super(RDC, self).__init__(alpha)
        self.fmx = fmx 
        self.fmy = fmy 

    def perform_test(self, pdata):
        with util.ContextTimer() as t:
            alpha = self.alpha

            rdf_stat, evals = self.compute_stat(pdata, return_eigvals=True)
            # the stat asymptotically follows a Chi^2
            Dx = self.fmx.num_features()
            Dy = self.fmy.num_features()
            pvalue = stats.chi2.sf(rdf_stat, df=Dx*Dy)

        results = {'alpha': self.alpha, 'pvalue': pvalue, 'test_stat': rdf_stat,
                'h0_rejected': pvalue < alpha, 'arr_eigvals': evals, 
                'time_secs': t.secs}
        return results

    def compute_stat(self, pdata, return_eigvals=False):
        X, Y = pdata.xy()
        n = pdata.sample_size()
        # copula transform to both X and Y
        cop_map = fea.MarginalCDFMap() 
        Xcdf = cop_map.gen_features(X)
        Ycdf = cop_map.gen_features(Y)

        # random Fourier features 
        Xrff = self.fmx.gen_features(Xcdf)
        Yrff = self.fmy.gen_features(Ycdf)

        # CCA 
        evals, Vx, Vy = util.cca(Xrff, Yrff)
        minD = min(Xrff.shape[1], Yrff.shape[1])
        # Barlett approximation 
        # Refer to https://en.wikipedia.org/wiki/Canonical_correlation
        bartlett_stat = -(n-1-0.5*(Xrff.shape[1]+Yrff.shape[1]+1))*np.sum(np.log(1-evals[:minD]**2))
        # Given in Lopez-Paz et al., 2013. Assume n_features is the same 
        # for both X and Y.
        #if self.fmx.num_features() != self.fmy.num_features():
        #  raise ValueError('Cannot use this Bartlett approximation when numbers of features of X and Y are different.')
        #bartlett_stat = ((self.fmx.num_features() + 3)/2.0 - n)*np.sum(np.log(1-evals**2))

        if return_eigvals:
            return bartlett_stat, evals
        else:
            return bartlett_stat

# end class RDC

class RDCPerm(IndTest):
    """
    An independece test with the Randomized Dependence Coefficient (Lopez-Paz et
    al., 2013). Use permutation to approximate the null distribution.
    The statistic is the canonical correlation of the random-feature transformed 
    data.
    """

    def __init__(self, fmx, fmy, n_permute=400, alpha=0.01, seed=27, use_copula=True):
        """
        fmx: a FeatureMap for the copula transform of X 
        fmy: a FeatureMap for the coputa transform of Y 
        use_copula: If False, do not use copula transform on the data.
        """
        super(RDCPerm, self).__init__(alpha)
        self.fmx = fmx 
        self.fmy = fmy 
        self.n_permute = n_permute
        self.seed = seed
        self.use_copula = use_copula

    def perform_test(self, pdata):
        with util.ContextTimer() as t:
            alpha = self.alpha
            rdc_stat, evals = self.compute_stat(pdata, return_eigvals=True)

            X, Y = pdata.xy()
            n_permute = self.n_permute
            arr_rdc = RDCPerm.list_permute(X, Y, self.fmx, self.fmy,
                    n_permute, seed=self.seed, use_copula=self.use_copula)
            # approximate p-value with the permutations 
            pvalue = np.mean(arr_rdc > rdc_stat)

        results = {'alpha': self.alpha, 'pvalue': pvalue, 'test_stat': rdc_stat,
                'h0_rejected': pvalue < alpha, 
                #'arr_rdc': arr_rdc, 
                'arr_eigvals': evals,
                'time_secs': t.secs, 'n_permute': n_permute}
        return results

    def compute_stat(self, pdata, return_eigvals=False):
        X, Y = pdata.xy()
        n = pdata.sample_size()
        if self.use_copula:
            # copula transform to both X and Y
            cop_map = fea.MarginalCDFMap() 
            Xtran = cop_map.gen_features(X)
            Ytran = cop_map.gen_features(Y)
        else:
            Xtran = X
            Ytran = Y

        # random Fourier features 
        Xrff = self.fmx.gen_features(Xtran) # n x Dx
        Yrff = self.fmy.gen_features(Ytran) # n x Dy

        # CCA 
        evals, Vx, Vy = util.cca(Xrff, Yrff, reg=1e-5)
        minD = min(Xrff.shape[1], Yrff.shape[1])
        if return_eigvals:
            return evals[0], evals[:minD]
        else:
            return evals[0]


    @staticmethod
    def list_permute(X, Y, fmx, fmy, n_permute=400, seed=8273, use_copula=True, cca_reg=1e-5):
        """
        Repeatedly mix, permute X, Y so that pairs are broken, and compute RDC.
        This is intended to be used to approximate the null distribution.

        Return a numpy array of RDC values
        """
        if X.shape[0] != Y.shape[0]:
            raise ValueError('X and Y must have the same number of rows (sample size)')
        n = X.shape[0]     
        arr = np.zeros(n_permute)
        if use_copula:
            # copula transform to both X and Y
            cop_map = fea.MarginalCDFMap() 
            Xtran = cop_map.gen_features(X)
            Ytran = cop_map.gen_features(Y)
        else:
            Xtran = X
            Ytran = Y
        # random Fourier features 
        Xf = fmx.gen_features(Xtran) # n x Dx
        Yf = fmy.gen_features(Ytran) # n x Dy
        Dx = Xf.shape[1]
        Dy = Yf.shape[1]            
        mx = np.mean(Xf, 0)
        my = np.mean(Yf, 0)
        mxmy = np.outer(mx, my)
        Cxx = np.cov(Xf.T)
        Cyy = np.cov(Yf.T)
        if Dx==1:
            CxxI = 1.0/Cxx
        else:   
            CxxI = np.linalg.inv(Cxx + cca_reg*np.eye(Dx))
        
        if Dy==1:
            CyyI = 1.0/Cyy
        else:
            CyyI = np.linalg.inv(Cyy + cca_reg*np.eye(Dy))
            
        with util.NumpySeedContext(seed=seed):
            r = 0 
            while r < n_permute:
                # shuffle the order of X, Y while still keeping the original pairs
                ind = np.random.choice(n, n, replace=False)
                Xfs = Xf[ind, :]
                Yfs = Yf[ind, :]

                # shift Yfs n-1 times 
                for s in range(n-1):
                    if r >= n_permute:
                        break
                    Yfs = np.roll(Yfs, 1, axis=0)
                    ### perform CCA
                    # Dx x Dy
                    Cxy = Xfs.T.dot(Yfs)/n - mxmy
                    CxxICxy = CxxI.dot(Cxy)
                    CyyICyx = CyyI.dot(Cxy.T)
                    
                    # problem for a
                    avals, aV = np.linalg.eig(CxxICxy.dot(CyyICyx))
                    # problem for b
                    #bvals, bV = np.linalg.eig(CyyICyx.dot(CxxICxy))
                    st = np.max(np.real(avals))
                    arr[r] = st
                    r = r + 1
        return arr

    @staticmethod 
    def _list_permute_naive(X, Y, fmx, fmy, n_permute=400, seed=8273, use_copula=True):
        """
        Repeatedly mix, permute X, Y so that pairs are broken, and compute RDC.
        This is intended to be used to approximate the null distribution.
        A naive implementation. 

        Return a numpy array of RDC values
        """
        pdata = data.PairedData(X, Y)
        rdc_perm = RDCPerm(fmx, fmy, n_permute=n_permute, use_copula=use_copula)
        n = pdata.sample_size()
        null_ps = data.PSNullResample(pdata)
        arr_rdc = np.zeros(n_permute)

        sub_seeds = util.subsample_ind(n_permute*2, n_permute, seed=seed)
        for i in range(n_permute):
            null_pdata = null_ps.sample(n, sub_seeds[i])
            stat = rdc_perm.compute_stat(null_pdata)
            arr_rdc[i] = stat 
        return arr_rdc
            

class GaussRDC(RDC):
    """
    An independence test with the Randomized Dependence Coefficient (Lopez-Paz et
    al., 2013) using Gaussian kernels for both X, Y. 

    H0: X and Y are independent 
    H1: X and Y are dependent.
    """

    def __init__(self, gwidthx, gwidthy, n_features_x, n_features_y,
            n_permute=400, alpha=0.01, seed=21):
        """
        :param gwidthx: Gaussian width^2 for the copula transforms of X
        :param n_features_x: The total number of features for x will be
            n_features_x*2.
        """
        fmx = fea.RFFKGauss(gwidthx, n_features_x, seed=seed)
        fmy = fea.RFFKGauss(gwidthy, n_features_y, seed=seed+2987)
        super(GaussRDC, self).__init__(fmx, fmy, n_permute, alpha)


####----- functions -------------####

def permute_null_dist(pdata, indtest, n_permute=400, seed=27):
    """
    Simulate from the null distribution by permuting the sample in the 
    PairedData pdata such that (x,y) pairs are broken. 
    indtest is an object with a method compute_stat(pdata).
    Making use of the structure of the independence test will likely result 
    in a more efficient permutations. This generic function is provided for
    convenience.

    Return an array of computed statistics.
    """
    n = pdata.sample_size()
    null_ps = data.PSNullResample(pdata)
    arr_stats = np.zeros(n_permute)

    sub_seeds = util.subsample_ind(100 + n_permute*2, n_permute, seed=seed)
    for i in range(n_permute):
        null_pdata = null_ps.sample(n, sub_seeds[i])
        stat = indtest.compute_stat(null_pdata)
        arr_stats[i] = stat 
    return arr_stats


def kl_kgauss_median(pdata):
    """
    Get two Gaussian kernels constructed with the median heuristic.
    """
    xtr, ytr = pdata.xy()
    dx = xtr.shape[1]
    dy = ytr.shape[1]
    medx2 = util.meddistance(xtr, subsample=1000)**2
    medy2 = util.meddistance(ytr, subsample=1000)**2
    # for classification problem, Y can be 0, 1. Subsampling in the computation 
    # of the median heuristic can yield 0.
    gwx2 = max(medx2, 1e-3)
    gwy2 = max(medy2, 1e-3)
    k = kernel.KGauss(gwx2)
    l = kernel.KGauss(gwy2)
    return k, l
