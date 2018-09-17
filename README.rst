.. -*- mode: rst -*-

|Travis|_ |Python27|_ |Python35|_ 

.. |Travis| image:: https://travis-ci.com/CherkaouiHamza/pybold.svg?token=tt8GRtf9hkYvmyTMbYvJ&branch=hcherkaoui/dev
.. _Travis: https://travis-ci.com/CherkaouiHamza/pybold
   
.. |Python27| image:: https://img.shields.io/badge/python-2.7-blue.svg
.. _Python27: https://badge.fury.io/py/scikit-learn

.. |Python35| image:: https://img.shields.io/badge/python-3.5-blue.svg
.. _Python35: https://badge.fury.io/py/scikit-learn


pyBOLD - dev
============

pyBOLD is a package module for fast and easy BOLD signal processing analysis.


Important links
===============

- Official source code repo: https://github.com/CherkaouiHamza/pybold


Dependencies
============

The required dependencies to use the software are:

* Psutil
* Joblib
* Numpy
* Scipy
* PyWavelets
* Matplotlib (for examples)


License
=======
All material is Free Software: BSD license (3 clause).


Installation
============

In order to perform the installation, run the following command from the pybold directory::

    python setup.py install --user

To run all the tests, run the following command from the pybold directory::

    python -m unittest discover pybold/tests

To run the examples, go to the directories examples and run a script, e.g.::

    python deconvolution_bloc.py


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
