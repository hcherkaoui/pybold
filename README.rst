.. -*- mode: rst -*-

pyBOLD
======

pyBOLD is a package module for fast and easy BOLD signal processing analysis.


Important links
===============

- Official source code repo: https://github.com/CherkaouiHamza/pybold


Dependencies
============

The required dependencies to use the software are:

* Python >= 2.7
* setuptools
* Numpy >= 1.8.2

If you are running the examples, matplotlib >= 1.3.1 is required.


Install
=======

In order to perform the installation, run the following command from the pybold directory::

    python setup.py install --user

To run all the tests, run the following command from the pybold directory::

    python -m unittest discover pybold/tests

To run the example, run the following command from the example directory::

    python blind_deconvolution.py

Development
===========

Code
----

GIT
~~~

You can check the latest sources with the command::

    git clone git://github.com/CherkaouiHamza/pybold

or if you have write privileges::

    git clone git@github.com:CherkaouiHamza/pybold
