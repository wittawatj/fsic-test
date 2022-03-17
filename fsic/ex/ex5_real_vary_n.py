"""
Experiments on various real datasets. Vary the sample sizes.
"""
__author__ = 'wittawat'


import fsic.data as data
import fsic.feature as fea
import fsic.indtest as it
import fsic.glo as glo
import fsic.util as util 
import fsic.kernel as kernel 
from . import exglobal as exglo

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
import logging
import numpy as np
import os
import re
import sys 
#import tempfile
import time


def job_nfsicJ10_stoopt(paired_source, tr, te, r, n_permute=None, J=10):
    k, l = kl_kgauss_median_bounds(tr)
    medx2 = k.sigma2
    medy2 = l.sigma2

    fac_min = 1e-1 
    fac_max = 5e3 

    with util.ContextTimer() as t:

        nfsic_opt_options = {'n_test_locs':J, 'max_iter':100,
        'V_step':1, 'W_step':1, 'gwidthx_step':1, 'gwidthy_step':1,
        'batch_proportion':1, 'tol_fun':1e-4, 'step_pow':0.5, 'seed':r+2,
        'reg': 1e-6, 
        'gwidthx_lb': max(1e-2, medx2*1e-3), 
        'gwidthx_ub': min(1e6, medx2*1e3), 
        'gwidthy_lb': max(1e-2, medy2*1e-3), 
        'gwidthy_ub': min(1e6, medy2*1e3),
        }
        op_V, op_W, op_gwx, op_gwy, info = it.GaussNFSIC.optimize_locs_widths(tr,
                alpha, **nfsic_opt_options )

        # make sure the optimized widths are not too extreme
        #last_gwx = info['gwidthxs'][-1]
        #last_gwy = info['gwidthys'][-1]
        #op_gwx = last_gwx
        #op_gwy = last_gwy
        op_gwx = max(fac_min*medx2, 1e-5, min(fac_max*medx2, op_gwx))
        op_gwy = max(fac_min*medy2, 1e-5, min(fac_max*medy2, op_gwy))

        nfsic_opt = it.GaussNFSIC(op_gwx, op_gwy, op_V, op_W, alpha=alpha,
                reg='auto', n_permute=n_permute, seed=r+3)
        nfsic_opt_result  = nfsic_opt.perform_test(te)
    return {
            # nfsic_opt's V, W can take up a lot of memory when d is huge.
            #'indtest': nfsic_opt,  
            'test_result': nfsic_opt_result, 'time_secs': t.secs}


def job_nfsicJ10_perm_stoopt(paired_source, tr, te, r):
    """
    Use permutations to simulate from the null distribution.
    """
    n_permute = 500
    return job_nfsicJ10_stoopt(paired_source, tr, te, r, n_permute)

def job_nfsicJ1_perm_stoopt(paired_source, tr, te, r):
    n_permute = 500
    J = 1
    return job_nfsicJ10_stoopt(paired_source, tr, te, r, n_permute, J)


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
        V, W = it.GaussNFSIC.init_locs_marginals_subset(pdata, J, seed=r+2)
        # May overfit and increase type-I errors?
        #V, W = it.GaussNFSIC.init_locs_joint_randn(pdata, J, seed=r+2)
        #with util.NumpySeedContext(seed=r+92):
        #   dx = pdata.dx()
        #   dy = pdata.dy()
        #   V = np.random.randn(J, dx)
        #   W = np.random.randn(J, dy)

        k, l = kl_kgauss_median_bounds(pdata)
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

    if pdata.sample_size() > 7000:
      # give up. Too big.
      k, l = kl_kgauss_median_bounds(pdata)
      qhsic = it.QuadHSIC(k, l, n_permute, alpha=alpha, seed=r+1)
      fake_result = {'alpha': alpha, 'pvalue': np.nan, 'test_stat': np.nan
              , 'h0_rejected': np.nan, 'time_secs': np.nan, 'n_permute': n_permute}
      return {'indtest': qhsic, 'test_result': fake_result, 'time_secs': np.nan}

    # Actually do the test
    with util.ContextTimer() as t:
        k, l = kl_kgauss_median_bounds(pdata)
        qhsic = it.QuadHSIC(k, l, n_permute, alpha=alpha, seed=r+1)
        qhsic_result = qhsic.perform_test(pdata)
    return {'indtest': qhsic, 'test_result': qhsic_result, 'time_secs': t.secs}


def job_fhsic_med(paired_source, tr, te, r):
    """
    HSIC with random Fourier features. Simulate the null distribution 
    with the spectrums of the empirical cross covariance operators.
    - Gaussian kernels.
    - No parameter selection procedure. Use the median heuristic for both 
    X and Y.
    - Use full sample for testing. 
    """
    
    n_simulate = 2000
    # random features 
    n_features = 10
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


def job_nyhsic_med(paired_source, tr, te, r):
    """
    HSIC with Nystrom approximation. Simulate the null distribution 
    with the spectrums of the empirical cross covariance operators.
    - Gaussian kernels.
    - No parameter selection procedure. Use the median heuristic for both 
    X and Y.
    - Use full sample for testing. 
    """
    
    n_simulate = 2000
    # random features 
    n_features = 10
    # use full sample for testing. Merge training and test sets
    pdata = tr + te
    with util.ContextTimer() as t:
        X, Y = pdata.xy()
        k, l = kl_kgauss_median_bounds(pdata)
        # randomly choose the inducing points from X, Y
        induce_x = util.subsample_rows(X, n_features, seed=r+2)
        induce_y = util.subsample_rows(Y, n_features, seed=r+3)

        nyhsic = it.NystromHSIC(k, l, induce_x, induce_y, n_simulate=n_simulate, alpha=alpha, seed=r+89)
        nyhsic_result = nyhsic.perform_test(pdata)
    return {
            #'indtest': nyhsic, 
            'test_result': nyhsic_result, 'time_secs': t.secs}


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

##-----------------------------------------------------------
def kl_kgauss_median_bounds(pdata):
    #print str(pdata)
    #print 'Y: '
    #print np.unique(pdata.Y, return_counts=True)

    k, l = it.kl_kgauss_median(pdata)

    logging.info('medx2: %g', k.sigma2)
    logging.info('medy2: %g', l.sigma2)
    # make sure that the widths are not too small. 
    k.sigma2 = max(k.sigma2, 1e-1)
    l.sigma2 = max(l.sigma2, 1e-1)
    return k, l

# Define our custom Job, which inherits from base class IndependentJob
class Ex5Job(IndependentJob):
   
    def __init__(self, aggregator, folder_path, prob_label, rep, job_func, n ):
        #walltime = 60*59*24 
        walltime = 60*59 
        memory = int(tr_proportion*n*1e-2) + 100

        IndependentJob.__init__(self, aggregator, walltime=walltime,
                               memory=memory)
        self.folder_path = folder_path
        self.prob_label = prob_label
        self.rep = rep
        self.job_func = job_func
        self.n = n

    # we need to define the abstract compute method. It has to return an instance
    # of JobResult base class
    def compute(self):
        
        # randomly wait a few seconds so that multiple processes accessing the same 
        # Theano function do not cause a lock problem. I do not know why.
        # I do not know if this does anything useful.
        # Sleep in seconds.
        time.sleep(np.random.rand(1)*2)

        # load the data and construct a PairedSource here 
        # The data can be big. We have to load it in this job function i.e., 
        # each computing node loads by itself (no data passing).
        folder_path = self.folder_path
        prob_label = self.prob_label
        paired_source, _, is_h0 = exglo.get_problem_pickle(folder_path, prob_label + '.n0')

        n = self.n
        r = self.rep
        job_func = self.job_func

        pdata = paired_source.sample(n, seed=r)
        with util.ContextTimer() as t: 
            logger.info("computing. %s. prob=%s, r=%d, n=%d"%(job_func.__name__,
                pdata.label, r, n))
            tr, te = pdata.split_tr_te(tr_proportion=tr_proportion, seed=r+21 )
            prob_label = self.prob_label

            job_result = job_func(paired_source, tr, te, r)

            # create ScalarResult instance
            result = SingleResult(job_result)
            # submit the result to my own aggregator
            self.aggregator.submit_result(result)
            func_name = job_func.__name__
        logger.info("done. ex1: %s, prob=%s, r=%d, n=%d. Took: %.3g s "%(func_name,
            pdata.label, r, n, t.secs))

        # save result
        fname = '%s-%s-r%d_n%d_a%.3f_trp%.2f.p' \
            %(prob_label, func_name, r, n, alpha, tr_proportion)
        glo.ex_save_result(ex, job_result, prob_label, fname)


# This import is needed so that pickle knows about the class Ex5Job.
# pickle is used when collecting the results from the submitted jobs.
from fsic.ex.ex5_real_vary_n import job_nfsicJ10_stoopt
from fsic.ex.ex5_real_vary_n import job_nfsicJ10_perm_stoopt
from fsic.ex.ex5_real_vary_n import job_nfsicJ1_perm_stoopt
from fsic.ex.ex5_real_vary_n import job_nfsicJ10_perm_med
from fsic.ex.ex5_real_vary_n import job_nfsicJ10_med
from fsic.ex.ex5_real_vary_n import job_qhsic_med
from fsic.ex.ex5_real_vary_n import job_nyhsic_med
from fsic.ex.ex5_real_vary_n import job_fhsic_med
from fsic.ex.ex5_real_vary_n import job_rdcperm_med
from fsic.ex.ex5_real_vary_n import Ex5Job


#--- experimental setting -----
ex = 5

alpha = 0.01
tr_proportion = 0.5

# number of trials
reps = 200

method_job_funcs = [ 
       job_nfsicJ10_stoopt,
       job_nfsicJ10_med,
       #job_nfsicJ10_perm_stoopt,
       #job_nfsicJ10_perm_med,
       job_qhsic_med, 
       job_nyhsic_med, 
       job_fhsic_med, 
       job_rdcperm_med,
       ]

# If is_rerun==False, do not rerun the experiment if a result file for the current
# setting already exists.
is_rerun = False
#---------------------------


def dataname2ns(data_name):
    very_low_ns = [200, 300, 400, 500]
    low_ns = [500, 1000, 1500, 2000, 5000]
    data2ns = { 
            'msd50000': [500, 1000, 1500, 2000],
            'data_n10000_td1878_vd2000': [2000, 4000, 6000, 8000],
            #
            'white_wine': very_low_ns,
            'news_popularity': low_ns,
            'skillcraft1': very_low_ns,
            'chromatic_music': very_low_ns,
            'music68': very_low_ns,
            'movie_rating': very_low_ns,
            'latlong_temp_y2013': low_ns,
            'voice_gender': very_low_ns,
            # Arizona feature selection data
            # http://featureselection.asu.edu/datasets.php
            'lung': [50, 100, 150], # lung max: 203
            'carcinom': [50, 100, 150], # carcinom max: 174
            'CLL_SUB_111': [50, 100 ], # max: 111
            'SMK_CAN_187': [50, 100, 150], #max: 187
            'TOX_171': [50, 100, 150], #max: 171
            #-------
            'higgs_no_deriv': very_low_ns,
            }
    if data_name not in data2ns:
        raise ValueError('Unknown data name: %s. Need to be one of %s'%(data_name, str(list(data2ns.keys())) ))
    return data2ns[data_name]


def run_problem(folder_path, prob_label):
    """Run the experiment"""

    pl = exglo.parse_prob_label(prob_label + '.n0')
    is_h0 = pl['is_h0']
    is_classification = pl['is_classification']
    data_name = pl['name']
    sample_sizes = dataname2ns(data_name)

    # ///////  submit jobs //////////
    # create folder name string
    #result_folder = glo.result_folder()
    #tmp_dir = tempfile.gettempdir()
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
    # repetitions x sample_sizes x #methods
    aggregators = np.empty((reps, len(sample_sizes), n_methods ), dtype=object)
    for r in range(reps):
        for ni, n in enumerate(sample_sizes):
            for mi, f in enumerate(method_job_funcs):
                # name used to save the result
                func_name = f.__name__
                fname = '%s-%s-r%d_n%d_a%.3f_trp%.2f.p' \
                    %(prob_label, func_name, r, n, alpha, tr_proportion)
                if not is_rerun and glo.ex_file_exists(ex, prob_label, fname):
                    logger.info('%s exists. Load and return.'%fname)
                    job_result = glo.ex_load_result(ex, prob_label, fname)

                    sra = SingleResultAggregator()
                    sra.submit_result(SingleResult(job_result))
                    aggregators[r, ni, mi] = sra
                else:
                    # result not exists or rerun
                    job = Ex5Job(SingleResultAggregator(), folder_path, prob_label, r,
                            f, n)
                    agg = engine.submit_job(job)
                    aggregators[r, ni, mi] = agg

    # let the engine finish its business
    logger.info("Wait for all call in engine")
    engine.wait_for_all()

    # ////// collect the results ///////////
    logger.info("Collecting results")
    job_results = np.empty((reps, len(sample_sizes), n_methods), dtype=object)
    for r in range(reps):
        for ni, n in enumerate(sample_sizes):
            for mi, f in enumerate(method_job_funcs):
                logger.info("Collecting result (%s, r=%d, n=%d)" %
                        (f.__name__, r, n))
                # let the aggregator finalize things
                aggregators[r, ni, mi].finalize()

                # aggregators[i].get_final_result() returns a SingleResult instance,
                # which we need to extract the actual result
                job_result = aggregators[r, ni, mi].get_final_result().result
                job_results[r, ni, mi] = job_result

    #func_names = [f.__name__ for f in method_job_funcs]
    #func2labels = exglobal.get_func2label_map()
    #method_labels = [func2labels[f] for f in func_names if f in func2labels]

    # save results 
    # - Do not store PairedSource because it can be very big.
    results = {'job_results': job_results, 'sample_sizes': sample_sizes,
            'is_h0': is_h0, 'is_classification': is_classification, 'alpha':
            alpha, 'repeats': reps, 'tr_proportion': tr_proportion,
            'method_job_funcs': method_job_funcs, 'prob_label': prob_label,  }
    
    # class name 
    fname = 'ex%d-%s-me%d_rs%d_nmi%d_nma%d_a%.3f_trp%.2f.p' \
        %(ex, prob_label, n_methods, reps, min(sample_sizes),
                max(sample_sizes), alpha, tr_proportion)
    glo.ex_save_result(ex, results, fname)
    logger.info('Saved aggregated results to %s'%fname)


def main():
    logging.getLogger().setLevel(logging.DEBUG)

    if len(sys.argv) != 3:
        print(('Usage: %s folder_path problem_label'%sys.argv[0]))
        print(' - folder_path: full path of the folder relative to fsic/data/ folder.')
        sys.exit(1)

    folder_path = sys.argv[1]
    prob_label = sys.argv[2]

    run_problem(folder_path, prob_label)

if __name__ == '__main__':
    main()

