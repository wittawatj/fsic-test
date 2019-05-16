
"""
A module containing functions shared among all experiments 
""" 

__author__ = 'wittawat'

import fsic.data as data
import fsic.feature as fea
import fsic.indtest as it
import fsic.glo as glo
import fsic.util as util 
import fsic.kernel as kernel 
import numpy as np
import os
import re

def get_func2label_map():
    # map: job_func_name |-> plot label
    #func_names = ['job_nfsic_opt', 'job_nfsic_grid']
    #labels = ['NFSIC-opt', 'NFSIC-grid' ]
    func_label_pairs = [
            ('job_nfsic_opt', 'NFSIC-opt'),
            ('job_nfsicJ3_opt', 'NFSICJ3-opt'),
            ('job_nfsicJ3_perm_stoopt', 'NFSICJ3P-sto'),
            ('job_nfsicJ3_s5_opt', 'NFSICJ3s5-opt'),
            ('job_nfsicJ10_opt', 'NFSICJ10-opt'),
            ('job_nfsicJ10_med', 'NFSIC-med'),
            ('job_nfsicJ10_perm_med', 'NFSIC-med'),
            ('job_nfsicJ10_stoopt', 'NFSIC-opt'),
            ('job_nfsicJ1_perm_stoopt', 'NFSICJ1P-sto'),
            ('job_nfsicJ10_perm_stoopt', 'NFSIC-opt'),
            ('job_nfsicJ10_cperm_stoopt', 'NFSICJ10CP-sto'),
            ('job_nfsic_grid', 'NFSIC-grid'),
            ('job_nfsic_med', 'NFSIC-med'),
            ('job_qhsic_med', 'QHSIC'),
            ('job_nyhsic_med', 'NyHSIC'),
            ('job_nyhsicD50_med', 'NyHSICD50-med'),
            ('job_fhsic_med', 'FHSIC'),
            ('job_fhsicD50_med', 'FHSICD50-med'),
            ('job_rdc_med', 'RDC-med'),
            ('job_rdcF50_med', 'RDCF50-med'),
            ('job_rdcperm_med', 'RDC'),
            ('job_rdcpermD50_med', 'RDCpD50-med'),
            ('job_rdcperm_nc_med', 'RDCp-nc-med'),
            ]
    #M = {k:v for (k,v) in zip(func_names, labels)}
    M = {k:v for (k,v) in func_label_pairs}
    return M

def func_plot_fmt_map():
    """
    Return a map from job function names to matplotlib plot styles 
    """
    # line_styles = ['o-', 'x-',  '*-', '-_', 'D-', 'h-', '+-', 's-', 'v-', 
    #               ',-', '1-']
    M = {}
    M['job_nfsic_opt'] = 'r-.'
    M['job_nfsicJ3_opt'] = 'rx-'
    M['job_nfsicJ3_perm_stoopt'] = 'rh:'
    M['job_nfsicJ3_s5_opt'] = 'mh:'
    M['job_nfsicJ10_opt'] = 'm^--'
    M['job_nfsicJ10_stoopt'] = 'rs-'
    M['job_nfsicJ1_perm_stoopt'] = 'm--'
    #M['job_nfsicJ10_perm_stoopt'] = 'ms-'
    M['job_nfsicJ10_perm_stoopt'] = 'rs-'
    M['job_nfsicJ10_cperm_stoopt'] = 'm*:'
    M['job_nfsic_grid'] = 'ro--'
    M['job_nfsic_med'] = 'r*--'
    M['job_nfsicJ10_med'] = 'rs-.' 
    M['job_nfsicJ10_perm_med'] = 'rs-.'

    M['job_qhsic_med'] = 'bo-'
    M['job_nyhsic_med'] = 'y*-'
    M['job_nyhsicD50_med'] = 'bx-.'
    M['job_fhsic_med'] = 'kD-'
    M['job_fhsicD50_med'] = 'k>--'
    M['job_rdc_med'] = 'gh-'

    M['job_rdcF50_med'] = 'g^--'
    M['job_rdcperm_med'] = 'g^-'
    M['job_rdcpermD50_med'] = 'g>--'
    M['job_rdcperm_nc_med'] = 'ms:'

    #M['job_scf_opt'] = 'r*-'
    #M['job_scf_opt10'] = 'r*-'
    #M['job_scf_gwgrid'] = 'r*--'

    #M['job_quad_mmd'] = 'g-^'
    #M['job_lin_mmd'] = 'cD-'
    #M['job_hotelling'] = 'yv-'
    return M


def parse_prob_label(label):
    """
    Parse the label into components. Currently 
    
    (name)_ndx3_ndy7_std_h0.n(sample_size)

    - The dot . separates the file name from options.
    - (name) can contain _.
    - (sample_size) must be less than the total size of the original data.
    - _ndx%d is to add %d dimensional standard Gaussian noise to X. Optional
      argument.
    - _ndy%d is to add %d dimensional standard Gaussian noise to Y. Optional
      argument.
    - _c (optional) to denote that Y is discrete (classification problem)
    - _std may or may not be present. If present, standardize the data in each 
        trial after resampling.
    - _h0 may or may not be present. If present, shuffle the sample to simulate 
        the case where H0 is true.

    Return a dictionary with keys and values.
    """
    pat = r'(\w+?)(_ndx(\d+))?(_ndy(\d+))?(_c)?(_std)?(_h0)?[.]n(\d+)'
    m = re.search(pat, label)
    if m is None:
        raise ValueError('prob_label "%s" does not follow the pattern "%s".'%(label, pat))

    fname = m.group(1)
    ndx = m.group(3) 
    ndx = 0 if ndx is None else int(ndx)

    ndy = m.group(5)
    ndy = 0 if ndy is None else int(ndy)

    is_classification = True if m.group(6) is not None else False
    is_std = True if m.group(7) is not None else False
    is_shuffled = True if m.group(8) is not None else False
    n = int(m.group(9))

    D = {'name': fname, 'ndx': ndx, 'ndy': ndy, 'is_classification':
            is_classification, 'is_std': is_std, 'is_h0': is_shuffled, 'n': n}
    return D


def get_problem_pickle(folder_path, prob_label):
    """
    - folder_path: path to a folder containing the data file relative to
      fsic/data/ folder.
    - prob_label: string of the form described in parse_prob_label() so that
      fsic/data/(folder_path)/(name).p exists.
        _n%d specifies the sample size to resample in each trial.

    Return a (PairedSource object, n, is_h0). 
    """
    dataset_dir = glo.data_file(folder_path)
    if not os.path.exists(dataset_dir):
        raise ValueError('dataset directory does not exist: %s'%dataset_dir)
    pl = parse_prob_label(prob_label)
    data_path = os.path.join(dataset_dir, pl['name']+'.p')
    if not os.path.exists(data_path):
        raise ValueError('dataset does not exist: %s'%data_path)

    loaded = glo.pickle_load(data_path)
    # Expected "loaded" to be a dictionary {'X': ..., 'Y': ..., ...}
    X, Y = loaded['X'], loaded['Y']

    is_h0 = pl['is_h0']
    is_c = pl['is_classification']
    if is_c:
        assert Y.shape[1]==1, 'Y should have one column. Shape = %s'%str(Y.shape)
        classes = Y[:, 0]
        # If the data is a classification problem, make a 1-of-K coding 
        # of the label. We assume that Y has one column.
        # modify Y 
        if  len(np.unique(Y))  > 2:
            # multiclass problem. Use 1-of-K coding.
            # Only for #classes > 2
            Y = util.one_of_K_code(classes)

    is_std = pl['is_std']
    n = pl['n']
    ndx = pl['ndx']
    ndy = pl['ndy']

    # Standardization after resampling can cause a 0 standard deviation. 
    # We will do it as the first step.
    #ps = data.PSStandardize(ps) if is_std else ps
    if is_std:
        X = util.standardize(X)
        Y = util.standardize(Y)

    pdata = data.PairedData(X, Y, label=prob_label)
    ps = data.PSStraResample(pdata, classes) if is_c else data.PSResample(pdata)
    ps = data.PSNullShuffle(ps) if is_h0 else ps
    if not (ndx == 0 and ndy == 0):
        ps = data.PSGaussNoiseDims(ps, ndx, ndy)
    return ps, n, is_h0




