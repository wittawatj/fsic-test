# The Finite Set Independence Criterion (FSIC)

[![Build Status](https://travis-ci.org/wittawatj/fsic-test.svg?branch=master)](https://travis-ci.org/wittawatj/fsic-test)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](https://github.com/wittawatj/fsic-test/blob/master/LICENSE)

This repository contains a Python 2.7 implementation of the normalized FSIC (NFSIC)
test as described in [our paper](https://arxiv.org/abs/1610.04782)

    An Adaptive Test of Independence with Analytic Kernel Embeddings
    Wittawat Jitkrittum, Zoltán Szabó, Arthur Gretton
    arXiv, 2016. 

## How to install?
1. Make sure that you have a complete [Scipy
   stack](https://www.scipy.org/stackspec.html) installed. One way to guarantee
this is to install it using [Anaconda with Python
2.7](https://www.continuum.io/downloads), which is also the environment we used
to develop this package. Make sure to use Python 2.7.
2. Clone or download this repository. You will get a folder with name `fsic-test`.
3. Add the path to the folder to Python's search path i.e., to `PYTHONPATH`
   global variable. See, for instance, [this page on
stackoverflow](http://stackoverflow.com/questions/11960602/how-to-add-something-to-pythonpath)
on how to do this in Linux. See
[here](http://stackoverflow.com/questions/3701646/how-to-add-to-the-pythonpath-in-windows-7)
for Windows. 
4. Check that indeed the package is in the search path by openning a new Python
   shell, and issuing `import fsic` (`fsic` is the name of our
Python package). If there is no import error, the installation is completed.  

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

