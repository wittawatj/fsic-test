# The Finite Set Independence Criterion (FSIC)

[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](https://github.com/wittawatj/fsic-test/blob/master/LICENSE)

This repository contains a Python 3 implementation of the normalized FSIC (NFSIC)
test as described in [our paper](https://arxiv.org/abs/1610.04782)

    An Adaptive Test of Independence with Analytic Kernel Embeddings
    Wittawat Jitkrittum, Zoltán Szabó, Arthur Gretton
    ICML 2017


## How to install?

If you plan to reproduce experimental results, you will probably want to modify
our code. It is best to install by:

1. Clone the repository by `git clone https://github.com/wittawatj/fsic-test`.
2. `cd` to the folder that you get, and install our package by

    pip install -e .

Alternatively, if you only want to use the developed package, you can do the
following without cloning the repository.

    pip install git+https://github.com/wittawatj/fsic-test.git

Either way, once installed, you should be able to do `import fsic` without any error.



### Dependency
We rely on the following Python packages during development.
Please make sure that you use the packages with the specified version numbers
or newer.

    numpy
    matplotlib
    scipy
    theano

Note that `theano` is not enabled in Anaconda by default. See [this
page](http://deeplearning.net/software/theano/install.html#basic-user-install-instructions)
for how to install it.

## Demo scripts

To get started, check
[demo_nfsic.ipynb](https://github.com/wittawatj/fsic-test/blob/master/ipynb/demo_nfsic.ipynb)
which will guide you through from the beginning. There are many Jupyter
notebooks in `ipynb` folder. Be sure to check them if you would like to explore more.


## Reproduce experimental results


### Experiments on test powers

All experiments which involve test powers on toy problems can be found in
`fsic/ex/ex1_vary_n.py`, `fsic/ex/ex2_prob_params.py`.
`fsic/ex/ex4_real_data.py`, and `fsic/ex/ex5_real_vary_n.py` are for
experiments on real data.  Each file is runnable with a command line argument.
For example in `ex1_vary_n.py`, we aim to check the test power of each test
as a function of the sample size `n`. The script `ex1_vary_n.py`
takes a dataset name as its argument. See `run_ex1.sh` which is a standalone
Bash script on how to execute  `ex1_power_vs_n.py`.

We used [independent-jobs](https://github.com/wittawatj/independent-jobs)
package to parallelize our experiments over a
[Slurm](http://slurm.schedmd.com/) cluster (the package is not needed if you
just need to use our developed tests). For example, for
`ex1_vary_n.py`, a job is created for each combination of

    (dataset, test algorithm, n, trial)

If you do not use Slurm, you can change the line

    engine = SlurmComputationEngine(batch_parameters)

to

    engine = SerialComputationEngine()

which will instruct the computation engine to just use a normal for-loop on a
single machine (will take a lot of time). Other computation engines that you
use might be supported. Running simulation will
create a lot of result files (one for each tuple above) saved as Pickle. Also, the `independent-jobs`
package requires a scratch folder to save temporary files for communication
among computing nodes.
Path to the folder containing the saved results (after running the experiments) is `fsic/result`.
Real data should be placed in `fsic/data`.


The scratch folder needed by the `independent-jobs` package can be specified in
`fsic/config.py`.  To plot the results, see the experiment's corresponding
Jupyter notebook in the `ipynb/` folder. For example, for `ex1_vary_n.py` see
`ipynb/ex1_results.ipynb` to plot the results.


## License
[MIT license](https://github.com/wittawatj/fsic-test/blob/master/LICENSE).

If you have questions or comments about anything regarding this work or code,
please do not hesitate to contact [Wittawat Jitkrittum](http://wittawat.com).

