"""Simulation to examine the P(reject) as the parameters for each problem are 
varied. What varies will depend on the problem."""

__author__ = 'wittawat'

import fsic.data as data
import fsic.feature as fea
import fsic.indtest as it
import fsic.glo as glo
import fsic.util as util 
import fsic.kernel as kernel 
from . import exglobal

# need independent_jobs package 
# https://github.com/karlnapf/independent-jobs
# The independent_jobs and fsic have to be in the global search path (.bashrc)
import independent_jobs as inj
from independent_jobs.jobs.IndependentJob import IndependentJob
from independent_jobs.results.SingleResult import SingleResult
from independent_jobs.aggregators.SingleResultAggregator import SingleResultAggregator
from independent_jobs.engines.BatchClusterParameters import BatchClusterParameters
from independent_jobs.engines.SerialComputationEngine import SerialComputationEngine
from independent_jobs.engines.SlurmComputationEngine import SlurmComputationEngine
from independent_jobs.tools.Log import logger
import math
import numpy as np
import os
import sys 
import time

"""
All the job functions return a dictionary with the following keys:
    - indtest: independence test object
    - test_result: the result from calling perform_test(te).
    - time_secs: run time in seconds 
"""

def job_nfsic_opt(paired_source, tr, te, r):
    """NFSIC with test locations optimzied.  """
    with util.ContextTimer() as t:
        nfsic_opt_options = {'n_test_locs':J, 'max_iter':200,
        'V_step':1, 'W_step':1, 'gwidthx_step':1, 'gwidthy_step':1,
        'batch_proportion':1.0, 'tol_fun':1e-4, 'step_pow':0.5, 'seed':r+2,
        'reg': 1e-6}
        op_V, op_W, op_gwx, op_gwy, info = it.GaussNFSIC.optimize_locs_widths(tr,
                alpha, **nfsic_opt_options )
        nfsic_opt = it.GaussNFSIC(op_gwx, op_gwy, op_V, op_W, alpha, reg='auto', seed=r+3)
        nfsic_opt_result  = nfsic_opt.perform_test(te)
    return {'indtest': nfsic_opt, 'test_result': nfsic_opt_result, 'time_secs': t.secs}


def job_nfsicJ3_opt(paired_source, tr, te, r, J=3):
    """NFSIC with test locations optimzied.  """
    with util.ContextTimer() as t:
        nfsic_opt_options = {'n_test_locs':J, 'max_iter':200,
        'V_step':1, 'W_step':1, 'gwidthx_step':1, 'gwidthy_step':1,
        'batch_proportion':1.0, 'tol_fun':1e-4, 'step_pow':0.5, 'seed':r+2,
        'reg': 1e-6}
        op_V, op_W, op_gwx, op_gwy, info = it.GaussNFSIC.optimize_locs_widths(tr,
                alpha, **nfsic_opt_options )
        nfsic_opt = it.GaussNFSIC(op_gwx, op_gwy, op_V, op_W, alpha, reg='auto', seed=r+3)
        nfsic_opt_result  = nfsic_opt.perform_test(te)
    return {'indtest':nfsic_opt, 'test_result': nfsic_opt_result, 'time_secs': t.secs}

def job_nfsicJ10_opt(paired_source, tr, te, r):
    return job_nfsicJ3_opt(paired_source, tr, te, r, J=10)

def job_nfsicJ10_stoopt(paired_source, tr, te, r, n_permute=None):
    J = 10
    with util.ContextTimer() as t:
        nfsic_opt_options = {'n_test_locs':J, 'max_iter':200,
        'V_step':1, 'W_step':1, 'gwidthx_step':1, 'gwidthy_step':1,
        'batch_proportion':0.7, 'tol_fun':1e-4, 'step_pow':0.5, 'seed':r+2,
        'reg': 1e-6}
        op_V, op_W, op_gwx, op_gwy, info = it.GaussNFSIC.optimize_locs_widths(tr,
                alpha, **nfsic_opt_options )
        nfsic_opt = it.GaussNFSIC(op_gwx, op_gwy, op_V, op_W, alpha, reg='auto', n_permute=n_permute, seed=r+3)
        nfsic_opt_result  = nfsic_opt.perform_test(te)
    return {
            #'indtest': nfsic_opt, 
            'test_result': nfsic_opt_result, 'time_secs': t.secs}


def job_nfsicJ10_perm_stoopt(paired_source, tr, te, r):
    """
    Use permutations to simulate from the null distribution.
    """
    n_permute = 500
    return job_nfsicJ10_stoopt(paired_source, tr, te, r, n_permute)

def job_nfsicJ3_perm_stoopt(paired_source, tr, te, r):
    """
    Use permutations to simulate from the null distribution.
    """
    n_permute = 500
    J = 3
    with util.ContextTimer() as t:
        nfsic_opt_options = {'n_test_locs':J, 'max_iter':300,
        'V_step':1, 'W_step':1, 'gwidthx_step':1, 'gwidthy_step':1,
        'batch_proportion':0.7, 'tol_fun':1e-4, 'step_pow':0.5, 'seed':r+2,
        'reg': 1e-6}
        op_V, op_W, op_gwx, op_gwy, info = it.GaussNFSIC.optimize_locs_widths(tr,
                alpha, **nfsic_opt_options )
        nfsic_opt = it.GaussNFSIC(op_gwx, op_gwy, op_V, op_W, alpha, reg='auto', n_permute=n_permute, seed=r+3)
        nfsic_opt_result  = nfsic_opt.perform_test(te)
    return {'indtest': nfsic_opt, 'test_result': nfsic_opt_result, 'time_secs': t.secs}

def job_nfsicJ10_cperm_stoopt(paired_source, tr, te, r):
    """
    - Copula transform the data
    - Use permutations to simulate from the null distribution.
    """
    n_permute = 500

    with util.ContextTimer() as t:
        # copula transform to both X and Y
        cop_map = fea.MarginalCDFMap() 
        xtr, ytr = tr.xy()
        xte, yte = te.xy()

        xtr = cop_map.gen_features(xtr)
        ytr = cop_map.gen_features(ytr)
        xte = cop_map.gen_features(xte)
        yte = cop_map.gen_features(yte)

        tr = data.PairedData(xtr, ytr)
        te = data.PairedData(xte, yte)

        to_return = job_nfsicJ10_stoopt(paired_source, tr, te, r, n_permute)
    to_return['time_secs'] = t.secs
    return to_return



def job_nfsicJ3_s5_opt(paired_source, tr, te, r):
    """NFSIC with test locations optimzied.  
    - Change step size to 5.
    - Test powers not very good.
    """
    J = 3
    with util.ContextTimer() as t:
        nfsic_opt_options = {'n_test_locs':J, 'max_iter':300,
        'V_step':5, 'W_step':5, 'gwidthx_step':1, 'gwidthy_step':1,
        'batch_proportion':1.0, 'tol_fun':1e-4, 'step_pow':0.5, 'seed':r+2,
        'reg': 1e-6}
        op_V, op_W, op_gwx, op_gwy, info = it.GaussNFSIC.optimize_locs_widths(tr,
                alpha, **nfsic_opt_options )
        nfsic_opt = it.GaussNFSIC(op_gwx, op_gwy, op_V, op_W, alpha, reg='auto', seed=r+3)
        nfsic_opt_result  = nfsic_opt.perform_test(te)
    return {'indtest': nfsic_opt, 'test_result': nfsic_opt_result, 'time_secs': t.secs}


def job_nfsic_grid(paired_source, tr, te, r):
    """
    NFSIC where the test locations are randomized, and the Gaussian widths 
    are optimized by a grid search.
    """
    # randomize the test locations by fitting Gaussians to the data
    with util.ContextTimer() as t:
        V, W = it.GaussNFSIC.init_locs_2randn(tr, J, seed=r+2)
        xtr, ytr = tr.xy()
        n_gwidth_cand = 30
        gwidthx_factors = 2.0**np.linspace(-4, 4, n_gwidth_cand) 
        gwidthy_factors = gwidthx_factors
        #gwidthy_factors = 2.0**np.linspace(-3, 4, 40) 
        medx = util.meddistance(xtr, 1000)
        medy = util.meddistance(ytr, 1000)
        list_gwidthx = np.hstack( ( (medx**2)*gwidthx_factors ) )
        list_gwidthy = np.hstack( ( (medy**2)*gwidthy_factors ) )

        bestij, lambs = it.GaussNFSIC.grid_search_gwidth(tr, V, W, list_gwidthx, list_gwidthy)
        # These are width^2
        best_widthx = list_gwidthx[bestij[0]]
        best_widthy = list_gwidthy[bestij[1]]

        # perform test
        nfsic_grid = it.GaussNFSIC(best_widthx, best_widthy, V, W, alpha)
        nfsic_grid_result = nfsic_grid.perform_test(te)
    return {'indtest': nfsic_grid, 'test_result': nfsic_grid_result, 'time_secs': t.secs}


def job_nfsic_med(paired_source, tr, te, r):
    """
    NFSIC in which the test locations are randomized, and the Gaussian width 
    is set with the median heuristic. Use full sample. No training/testing splits.
    """
    pdata = tr + te
    with util.ContextTimer() as t:
        V, W = it.GaussNFSIC.init_locs_2randn(pdata, J, seed=r+2)
        k, l = kl_kgauss_median(pdata)
        nfsic_med = it.NFSIC(k, l, V, W, alpha=alpha, reg='auto')
        nfsic_med_result = nfsic_med.perform_test(pdata)
    return {
            #'indtest': nfsic_med, 
            'test_result': nfsic_med_result, 'time_secs': t.secs}


def job_nfsicJ10_med(paired_source, tr, te, r, n_permute=None):
    """
    NFSIC in which the test locations are randomized, and the Gaussian width 
    is set with the median heuristic. Use full sample. No training/testing splits.
    J=10
    """
    J = 10
    pdata = tr + te
    with util.ContextTimer() as t:
        #V, W = it.GaussNFSIC.init_locs_2randn(pdata, J, seed=r+2)
        # May overfit and increase type-I errors?
        #V, W = it.GaussNFSIC.init_locs_marginals_subset(pdata, J, seed=r+2)
        V, W = it.GaussNFSIC.init_locs_joint_randn(pdata, J, seed=r+2)
        #with util.NumpySeedContext(seed=r+92):
        #    dx = pdata.dx()
        #    dy = pdata.dy()
        #    V = np.random.randn(J, dx)
        #    W = np.random.randn(J, dy)
        k, l = kl_kgauss_median(pdata)

        nfsic_med = it.NFSIC(k, l, V, W, alpha=alpha, reg='auto',
                n_permute=n_permute, seed=r+3)
        nfsic_med_result = nfsic_med.perform_test(pdata)
    return {
            #'indtest': nfsic_med, 
            'test_result': nfsic_med_result, 'time_secs': t.secs}


def job_nfsicJ10_perm_med(paired_source, tr, te, r):
    n_permute = 500
    return job_nfsicJ10_med(paired_source, tr, te, r, n_permute=n_permute)


def job_qhsic_med(paired_source, tr, te, r):
    """
    Quadratic-time HSIC using the permutation test.
    - Gaussian kernels.
    - No parameter selection procedure. Use the median heuristic for both 
    X and Y.
    - Use full sample for testing. 
    """
    # use full sample for testing. Merge training and test sets
    pdata = tr + te
    n_permute = 300
    with util.ContextTimer() as t:
        k, l = kl_kgauss_median(pdata)
        qhsic = it.QuadHSIC(k, l, n_permute, alpha=alpha, seed=r+1)
        qhsic_result = qhsic.perform_test(pdata)
    return {'indtest': qhsic, 'test_result': qhsic_result, 'time_secs': t.secs}


def job_fhsic_med(paired_source, tr, te, r, n_features=10):
    """
    HSIC with random Fourier features. Simulate the null distribution 
    with the spectrums of the empirical cross covariance operators.
    - Gaussian kernels.
    - No parameter selection procedure. Use the median heuristic for both 
    X and Y.
    - Use full sample for testing. 
    - n_features: number of random features
    """
    
    n_simulate = 2000
    # use full sample for testing. Merge training and test sets
    pdata = tr + te
    with util.ContextTimer() as t:
        X, Y = pdata.xy()
        medx = util.meddistance(X, subsample=1000)
        medy = util.meddistance(Y, subsample=1000)
        sigmax2 = medx**2
        sigmay2 = medy**2

        fmx = fea.RFFKGauss(sigmax2, n_features=n_features, seed=r+1)
        fmy = fea.RFFKGauss(sigmay2, n_features=n_features, seed=r+2)
        ffhsic = it.FiniteFeatureHSIC(fmx, fmy, n_simulate=n_simulate, alpha=alpha, seed=r+89)
        ffhsic_result = ffhsic.perform_test(pdata)
    return {'indtest': ffhsic, 'test_result': ffhsic_result, 'time_secs': t.secs}

def job_fhsicD50_med(paired_source, tr, te, r):
    return job_fhsic_med(paired_source, tr, te, r, n_features=50)


def job_nyhsic_med(paired_source, tr, te, r, n_features=10):
    """
    HSIC with Nystrom approximation. Simulate the null distribution 
    with the spectrums of the empirical cross covariance operators.
    - Gaussian kernels.
    - No parameter selection procedure. Use the median heuristic for both 
    X and Y.
    - Use full sample for testing. 
    - n_features: number of randomly selected points for basis 
    """
    
    n_simulate = 2000
    # use full sample for testing. Merge training and test sets
    pdata = tr + te
    with util.ContextTimer() as t:
        X, Y = pdata.xy()
        k, l =kl_kgauss_median(pdata)
        # randomly choose the inducing points from X, Y
        induce_x = util.subsample_rows(X, n_features, seed=r+2)
        induce_y = util.subsample_rows(Y, n_features, seed=r+3)

        nyhsic = it.NystromHSIC(k, l, induce_x, induce_y, n_simulate=n_simulate, alpha=alpha, seed=r+89)
        nyhsic_result = nyhsic.perform_test(pdata)
    return {'indtest': nyhsic, 'test_result': nyhsic_result, 'time_secs': t.secs}

def job_nyhsicD50_med(paired_source, tr, te, r):
    return job_nyhsic_med(paired_source, tr, te, r, n_features=50)

def job_rdcperm_med(paired_source, tr, te, r, n_features=10):
    """
    The Randomized Dependence Coefficient test with permutations.
    """
    pdata = tr + te 
    n_permute = 500
    # n_features=10 from Lopez-Paz et al., 2013 paper.
    with util.ContextTimer() as t:
        # get the median distances 
        X, Y = pdata.xy()
        # copula transform to both X and Y
        cop_map = fea.MarginalCDFMap() 
        Xcdf = cop_map.gen_features(X)
        Ycdf = cop_map.gen_features(Y)

        medx = util.meddistance(Xcdf, subsample=1000)
        medy = util.meddistance(Ycdf, subsample=1000)
        sigmax2 = medx**2
        sigmay2 = medy**2

        fmx = fea.RFFKGauss(sigmax2, n_features=n_features, seed=r+19)
        fmy = fea.RFFKGauss(sigmay2, n_features=n_features, seed=r+220)
        rdcperm = it.RDCPerm(fmx, fmy, n_permute=n_permute, alpha=alpha, seed=r+100)
        rdcperm_result = rdcperm.perform_test(pdata)
    return {'indtest': rdcperm, 'test_result': rdcperm_result, 'time_secs': t.secs}

def job_rdcpermD50_med(paired_source, tr, te, r):
    return job_rdc_med(paired_source, tr, te, r, n_features=50)

def job_rdcperm_nc_med(paired_source, tr, te, r, n_features=10):
    """
    The Randomized Dependence Coefficient test with permutations.
    No copula transformtation. Use median heuristic on the data.
    """
    pdata = tr + te 
    n_permute = 500
    # n_features=10 from Lopez-Paz et al., 2013 paper.
    with util.ContextTimer() as t:
        # get the median distances 
        X, Y = pdata.xy()

        medx = util.meddistance(X, subsample=1000)
        medy = util.meddistance(Y, subsample=1000)
        sigmax2 = medx**2
        sigmay2 = medy**2

        fmx = fea.RFFKGauss(sigmax2, n_features=n_features, seed=r+19)
        fmy = fea.RFFKGauss(sigmay2, n_features=n_features, seed=r+220)
        rdcperm = it.RDCPerm(fmx, fmy, n_permute=n_permute, alpha=alpha,
                seed=r+100, use_copula=False)
        rdcperm_result = rdcperm.perform_test(pdata)
    return {'indtest': rdcperm, 'test_result': rdcperm_result, 'time_secs': t.secs}


def job_rdc_med(paired_source, tr, te, r, n_features=10):
    """
    The Randomized Dependence Coefficient test.
    - Gaussian width = median heuristic on the copula-transformed data 
    - 10 random features for each X andY
    - Use full dataset for testing
    """
    pdata = tr + te 
    # n_features=10 from Lopez-Paz et al., 2013 paper.
    with util.ContextTimer() as t:
        # get the median distances 
        X, Y = pdata.xy()
        # copula transform to both X and Y
        cop_map = fea.MarginalCDFMap() 
        Xcdf = cop_map.gen_features(X)
        Ycdf = cop_map.gen_features(Y)

        medx = util.meddistance(Xcdf, subsample=1000)
        medy = util.meddistance(Ycdf, subsample=1000)
        sigmax2 = medx**2
        sigmay2 = medy**2

        fmx = fea.RFFKGauss(sigmax2, n_features=n_features, seed=r+19)
        fmy = fea.RFFKGauss(sigmay2, n_features=n_features, seed=r+220)
        rdc = it.RDC(fmx, fmy, alpha=alpha)
        rdc_result = rdc.perform_test(pdata)
    return {'indtest': rdc, 'test_result': rdc_result, 'time_secs': t.secs}

def job_rdcF50_med(paired_source, tr, te, r):
    """
    RDC with 50 features. 
    """
    return job_rdc_med(paired_source, tr, te, r, n_features=50)


##-----------------------------------------------------------
def kl_kgauss_median(pdata):
    """
    Get two Gaussian kernels constructed with the median heuristic.
    """
    xtr, ytr = pdata.xy()
    dx = xtr.shape[1]
    dy = ytr.shape[1]
    medx2 = util.meddistance(xtr, subsample=1000)**2
    medy2 = util.meddistance(ytr, subsample=1000)**2
    k = kernel.KGauss(medx2)
    l = kernel.KGauss(medy2)
    return k, l

# Define our custom Job, which inherits from base class IndependentJob
class Ex2Job(IndependentJob):
   
    def __init__(self, aggregator, paired_source, prob_label, rep, job_func,
            prob_param):
        #walltime = 60*59*24 
        walltime = 60*59
        memory = int(tr_proportion*sample_size*1e-2) + 50

        IndependentJob.__init__(self, aggregator, walltime=walltime,
                               memory=memory)
        self.paired_source = paired_source
        self.prob_label = prob_label
        self.rep = rep
        self.job_func = job_func
        self.prob_param = prob_param

    # we need to define the abstract compute method. It has to return an instance
    # of JobResult base class
    def compute(self):
        
        # randomly wait a few seconds so that multiple processes accessing the same 
        # Theano function do not cause a lock problem. I do not know why.
        # I do not know if this does anything useful.
        # Sleep in seconds.
        time.sleep(np.random.rand(1)*3)

        paired_source = self.paired_source 
        r = self.rep
        prob_param = self.prob_param
        job_func = self.job_func
        # sample_size is a global variable
        pdata = paired_source.sample(sample_size, seed=r)
        with util.ContextTimer() as t :
            logger.info("computing. %s. prob=%s, r=%d, param=%.3g"%(job_func.__name__, pdata.label, r, prob_param))

            tr, te = pdata.split_tr_te(tr_proportion=tr_proportion, seed=r+21 )
            prob_label = self.prob_label

            job_result = job_func(paired_source, tr, te, r)

            # create ScalarResult instance
            result = SingleResult(job_result)
            # submit the result to my own aggregator
            self.aggregator.submit_result(result)
            func_name = job_func.__name__
        logger.info("done. ex2: %s, prob=%s, r=%d, param=%.3g. Took: %.3g s "%(func_name,
            pdata.label, r, prob_param, t.secs))

        # save result
        fname = '%s-%s-n%d_J%d_r%d_p%.3f_a%.3f_trp%.2f.p' \
                %(prob_label, func_name, sample_size, J, r, prob_param, alpha,
                        tr_proportion)
        glo.ex_save_result(ex, job_result, prob_label, fname)


# This import is needed so that pickle knows about the class Ex2Job.
# pickle is used when collecting the results from the submitted jobs.
from fsic.ex.ex2_prob_params import job_nfsic_opt
from fsic.ex.ex2_prob_params import job_nfsicJ3_opt
from fsic.ex.ex2_prob_params import job_nfsicJ10_opt
from fsic.ex.ex2_prob_params import job_nfsicJ10_stoopt
from fsic.ex.ex2_prob_params import job_nfsicJ10_perm_stoopt
from fsic.ex.ex2_prob_params import job_nfsicJ3_perm_stoopt
from fsic.ex.ex2_prob_params import job_nfsicJ10_cperm_stoopt
from fsic.ex.ex2_prob_params import job_nfsicJ3_s5_opt
from fsic.ex.ex2_prob_params import job_nfsic_grid
from fsic.ex.ex2_prob_params import job_nfsic_med
from fsic.ex.ex2_prob_params import job_nfsicJ10_med
from fsic.ex.ex2_prob_params import job_nfsicJ10_perm_med
from fsic.ex.ex2_prob_params import job_qhsic_med
from fsic.ex.ex2_prob_params import job_nyhsic_med
from fsic.ex.ex2_prob_params import job_nyhsicD50_med
from fsic.ex.ex2_prob_params import job_fhsic_med
from fsic.ex.ex2_prob_params import job_fhsicD50_med
from fsic.ex.ex2_prob_params import job_rdc_med
from fsic.ex.ex2_prob_params import job_rdcperm_med
from fsic.ex.ex2_prob_params import job_rdcpermD50_med
from fsic.ex.ex2_prob_params import job_rdcperm_nc_med
from fsic.ex.ex2_prob_params import job_rdcF50_med
from fsic.ex.ex2_prob_params import Ex2Job


#--- experimental setting -----
ex = 2

# sample size = n (the training and test sizes are n/2)
sample_size = 4000

# number of test locations / test frequencies J
J = 1
alpha = 0.05
tr_proportion = 0.5
# repetitions for each parameter setting
reps = 200
#method_job_funcs = [job_nfsic_opt, job_nfsicJ3_opt, job_nfsicJ10_opt,
#        job_nfsicJ10_stoopt, job_nfsicJ3_s5_opt, job_nfsic_med, job_qhsic_med,
#        job_rdcperm_med ]

#method_job_funcs = [ 
#        job_nfsicJ10_stoopt,
#        ]
method_job_funcs = [ 
       #job_nfsicJ3_perm_stoopt, 
       job_nfsicJ10_stoopt,
       job_nfsicJ10_med,

       #job_nfsicJ10_perm_stoopt,
       #job_nfsicJ10_perm_med,

       job_qhsic_med, 
       job_nyhsic_med, 
       #job_nyhsicD50_med,
       job_fhsic_med, 
       #job_fhsicD50_med,
       job_rdcperm_med,
       #job_rdcperm_nc_med,
       #job_rdcpermD50_med,

       ]

# If is_rerun==False, do not rerun the experiment if a result file for the current
# setting of (pi, r) already exists.
is_rerun = False
#---------------------------

def get_paired_source_list(prob_label):
    """Return (prob_params, ps_list) where 
    - ps_list: a list of PairedSource's representing the problems, each 
    corresponding to one parameter setting.
    - prob_params: the list of problem parameters. Each parameter has to be a
      scalar (so that we can plot them later). Parameters are preferably
      positive integers.
    """
    # map: prob_label -> [paired_source]
    degrees = [float(deg) for deg in range(0, 10+1, 2)]
    noise_dims = list(range(0, 8, 2))
    sg_dims = list(range(10, 100, 20))
    # Medium-sized Gaussian problem 
    msg_dims = list(range(50, 250+1, 50))
    # Big Gaussian problem
    bsg_dims = list(range(100, 400+1, 100))
    sin_freqs = list(range(1, 6+1))
    multi_sin_d = list(range(1, 4+1, 1))
    pwsign_d = list(range(10, 50+1, 10))
    gauss_sign_d = list(range(1, 6+1, 1))
    prob2ps = { 
            'u2drot': (degrees, 
                [data.PS2DUnifRotate(2.0*np.pi*deg/360, xlb=-1,
                xub=1, ylb=-1, yub=1) for deg in degrees]
                ),
            'urot_noise' : (noise_dims, 
                [data.PSUnifRotateNoise(angle=np.pi/4, xlb=-1, xub=1, ylb=-1, yub=1,
                    noise_dim=d) for d in noise_dims ]
                ),
            'sg': (sg_dims, 
                [data.PSIndSameGauss(dx=d, dy=d) for d in sg_dims]
                ), 
            'msg': (msg_dims, 
                [data.PSIndSameGauss(dx=d, dy=d) for d in msg_dims]
                ), 
            'bsg': (bsg_dims, 
                [data.PSIndSameGauss(dx=d, dy=d) for d in bsg_dims]
                ), 
            'sin': (sin_freqs, 
                [data.PS2DSinFreq(freq=f) for f in sin_freqs]), 
            'msin': (multi_sin_d, 
                [data.PSSinFreq(freq=1, d=d) for d in multi_sin_d]),

            'pwsign': (pwsign_d, 
                [data.PSPairwiseSign(dx=d) for d in pwsign_d]),

            'gsign': (gauss_sign_d, 
                [data.PSGaussSign(dx=d) for d in gauss_sign_d]),
            }
    if prob_label not in prob2ps:
        raise ValueError('Unknown problem label. Need to be one of %s'%str(list(prob2ps.keys())) )
    return prob2ps[prob_label]



def run_problem(prob_label):
    """Run the experiment"""
    prob_params, list_ps = get_paired_source_list(prob_label)

    # ///////  submit jobs //////////
    # create folder name string
    #result_folder = glo.result_folder()
    from fsic.config import expr_configs
    tmp_dir = expr_configs['scratch_dir']
    foldername = os.path.join(tmp_dir, 'wj_slurm', 'e%d'%ex)
    logger.info("Setting engine folder to %s" % foldername)

    # create parameter instance that is needed for any batch computation engine
    logger.info("Creating batch parameter instance")
    batch_parameters = BatchClusterParameters(
        foldername=foldername, job_name_base="e%d_"%ex, parameter_prefix="")

    # Use the following line if Slurm queue is not used.
    #engine = SerialComputationEngine()
    engine = SlurmComputationEngine(batch_parameters)
    n_methods = len(method_job_funcs)
    # repetitions x len(prob_params) x #methods
    aggregators = np.empty((reps, len(prob_params), n_methods ), dtype=object)
    for r in range(reps):
        for pi, param in enumerate(prob_params):
            for mi, f in enumerate(method_job_funcs):
                # name used to save the result
                func_name = f.__name__
                fname = '%s-%s-n%d_J%d_r%d_p%.3f_a%.3f_trp%.2f.p' \
                    %(prob_label, func_name, sample_size, J, r, param, alpha,
                            tr_proportion)
                if not is_rerun and glo.ex_file_exists(ex, prob_label, fname):
                    logger.info('%s exists. Load and return.'%fname)
                    job_result = glo.ex_load_result(ex, prob_label, fname)

                    sra = SingleResultAggregator()
                    sra.submit_result(SingleResult(job_result))
                    aggregators[r, pi, mi] = sra
                else:
                    # result not exists or rerun
                    job = Ex2Job(SingleResultAggregator(), list_ps[pi],
                            prob_label, r, f, param)
                    agg = engine.submit_job(job)
                    aggregators[r, pi, mi] = agg

    # let the engine finish its business
    logger.info("Wait for all call in engine")
    engine.wait_for_all()

    # ////// collect the results ///////////
    logger.info("Collecting results")
    job_results = np.empty((reps, len(prob_params), n_methods), dtype=object)
    for r in range(reps):
        for pi, param in enumerate(prob_params):
            for mi, f in enumerate(method_job_funcs):
                logger.info("Collecting result (%s, r=%d, param=%.3g)" %
                        (f.__name__, r, param))
                # let the aggregator finalize things
                aggregators[r, pi, mi].finalize()

                # aggregators[i].get_final_result() returns a SingleResult instance,
                # which we need to extract the actual result
                job_result = aggregators[r, pi, mi].get_final_result().result
                job_results[r, pi, mi] = job_result

    #func_names = [f.__name__ for f in method_job_funcs]
    #func2labels = exglobal.get_func2label_map()
    #method_labels = [func2labels[f] for f in func_names if f in func2labels]

    # save results 
    results = {'job_results': job_results, 'prob_params': prob_params, 
            'alpha': alpha, 'J': J, 'repeats': reps, 'list_paired_source': list_ps, 
            'tr_proportion': tr_proportion, 'method_job_funcs': method_job_funcs, 
            'prob_label': prob_label, 'sample_size': sample_size, 
            }
    
    # class name 
    fname = 'ex%d-%s-me%d_n%d_J%d_rs%d_pmi%.3f_pma%.3f_a%.3f_trp%.2f.p' \
        %(ex, prob_label, n_methods, sample_size, J, reps, min(prob_params),
                max(prob_params), alpha, tr_proportion)
    glo.ex_save_result(ex, results, fname)
    logger.info('Saved aggregated results to %s'%fname)


def main():
    if len(sys.argv) != 2:
        print(('Usage: %s problem_label'%sys.argv[0]))
        sys.exit(1)
    prob_label = sys.argv[1]

    run_problem(prob_label)

if __name__ == '__main__':
    main()

