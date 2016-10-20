"""Module containing convenient functions for plotting"""

__author__ = 'wittawat'

import fsic.ex.exglobal as exglo
import fsic.glo as glo
import matplotlib.pyplot as plt
import numpy as np

def plot_2d_data(pdata):
    """
    pdata: an instance of PairedData
    Return a figure handle
    """
    X, Y = pdata.xy()
    n, d = X.shape 
    if d != 2:
        raise ValueError('d must be 2 to plot.') 
    # plot
    fig = plt.figure()
    plt.plot(X, Y, 'ob')
    plt.title(pdata.label)
    return fig


def plot_prob_reject(ex, fname, h1_true, func_xvalues, xlabel,
        func_title=None):
    """
    plot the empirical probability that the statistic is above the threshold.
    This can be interpreted as type-1 error (when H0 is true) or test power 
    (when H1 is true). The plot is against the specified x-axis.

    - ex: experiment number 
    - fname: file name of the aggregated result
    - h1_true: True if H1 is true 
    - func_xvalues: function taking aggregated results dictionary and return the values 
        to be used for the x-axis values.            
    - xlabel: label of the x-axis. 
    - func_title: a function: results dictionary -> title of the plot

    Return loaded results
    """
    #from IPython.core.debugger import Tracer 
    #Tracer()()

    results = glo.ex_load_result(ex, fname)

    def rej_accessor(jr):
        rej = jr['test_result']['h0_rejected']
        # When used with vectorize(), making the value float will make the resulting 
        # numpy array to be of float. nan values can be stored.
        return float(rej)

    #value_accessor = lambda job_results: job_results['test_result']['h0_rejected']
    vf_pval = np.vectorize(rej_accessor)
    # results['job_results'] is a dictionary: 
    # {'test_result': (dict from running perform_test(te) '...':..., }
    rejs = vf_pval(results['job_results'])
    repeats, _, n_methods = results['job_results'].shape
    mean_rejs = np.mean(rejs, axis=0)
    #print mean_rejs
    #std_pvals = np.std(rejs, axis=0)
    #std_pvals = np.sqrt(mean_rejs*(1.0-mean_rejs))

    xvalues = func_xvalues(results)

    #ns = np.array(results[xkey])
    #te_proportion = 1.0 - results['tr_proportion']
    #test_sizes = ns*te_proportion
    line_styles = exglo.func_plot_fmt_map()
    method_labels = exglo.get_func2label_map()
    
    func_names = [f.__name__ for f in results['method_job_funcs'] ]
    for i in range(n_methods):    
        te_proportion = 1.0 - results['tr_proportion']
        fmt = line_styles[func_names[i]]
        #plt.errorbar(ns*te_proportion, mean_rejs[:, i], std_pvals[:, i])
        method_label = method_labels[func_names[i]]
        plt.plot(xvalues, mean_rejs[:, i], fmt, label=method_label)
    '''
    else:
        # h0 is true 
        z = stats.norm.isf( (1-confidence)/2.0)
        for i in range(n_methods):
            phat = mean_rejs[:, i]
            conf_iv = z*(phat*(1-phat)/repeats)**0.5
            #plt.errorbar(test_sizes, phat, conf_iv, fmt=line_styles[i], label=method_labels[i])
            plt.plot(test_sizes, mean_rejs[:, i], line_styles[i], label=method_labels[i])
    '''
            
    ylabel = 'Test power' if h1_true else 'Type-I error'
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.xticks( np.hstack((xvalues) ))
    
    alpha = results['alpha']
    plt.legend(loc='best')
    title = '%s. %d trials. $\\alpha$ = %.2g.'%( results['prob_label'],
            repeats, alpha) if func_title is None else func_title(results)
    plt.title(title)
    #plt.grid()
    return results
        

def plot_runtime(ex, fname, func_xvalues, xlabel, func_title=None):
    results = glo.ex_load_result(ex, fname)
    value_accessor = lambda job_results: job_results['time_secs']
    vf_pval = np.vectorize(value_accessor)
    # results['job_results'] is a dictionary: 
    # {'test_result': (dict from running perform_test(te) '...':..., }
    times = vf_pval(results['job_results'])
    repeats, _, n_methods = results['job_results'].shape
    time_avg = np.mean(times, axis=0)
    time_std = np.std(times, axis=0)

    xvalues = func_xvalues(results)

    #ns = np.array(results[xkey])
    #te_proportion = 1.0 - results['tr_proportion']
    #test_sizes = ns*te_proportion
    line_styles = exglo.func_plot_fmt_map()
    method_labels = exglo.get_func2label_map()
    
    func_names = [f.__name__ for f in results['method_job_funcs'] ]
    for i in range(n_methods):    
        te_proportion = 1.0 - results['tr_proportion']
        fmt = line_styles[func_names[i]]
        #plt.errorbar(ns*te_proportion, mean_rejs[:, i], std_pvals[:, i])
        method_label = method_labels[func_names[i]]
        plt.errorbar(xvalues, time_avg[:, i], yerr=time_std[:,i], fmt=fmt,
                label=method_label)
            
    ylabel = 'Time (s)'
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.xlim([np.min(xvalues), np.max(xvalues)])
    plt.xticks( xvalues, xvalues )
    plt.legend(loc='best')
    plt.gca().set_yscale('log')
    title = '%s. %d trials. '%( results['prob_label'],
            repeats ) if func_title is None else func_title(results)
    plt.title(title)
    #plt.grid()
    return results


