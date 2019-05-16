# The Finite Set Independence Criterion (FSIC)

[![Build Status](https://travis-ci.org/wittawatj/fsic-test.svg?branch=master)](https://travis-ci.org/wittawatj/fsic-test)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](https://github.com/wittawatj/fsic-test/blob/master/LICENSE)

This repository contains a Python 2.7 implementation of the normalized FSIC (NFSIC)
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

    numpy==1.11.0
    matplotlib==1.5.1
    scipy==0.18.0
    theano==0.8.

Note that `theano` is not enabled in Anaconda by default. See [this
page](http://deeplearning.net/software/theano/install.html#basic-user-install-instructions)
for how to install it.

## Demo scripts

To get started, check
[demo_nfsic.ipynb](https://github.com/wittawatj/fsic-test/blob/master/ipynb/demo_nfsic.ipynb)
which will guide you through from the beginning. There are many Jupyter
notebooks in `ipynb` folder. Be sure to check them if you would like to explore more.

## License
[MIT license](https://github.com/wittawatj/fsic-test/blob/master/LICENSE).

If you have questions or comments about anything regarding this work or code,
please do not hesitate to contact [Wittawat Jitkrittum](http://wittawat.com).

