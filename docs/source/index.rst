.. pybear documentation master file, created by
   sphinx-quickstart on Mon Jun 30 09:26:37 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the pybear documentation
===================================

pybear is a scikit-learn-style Python computing library that augments data 
analytics functionality found in popular packages like scikit-learn and xgboost.

pybear currently supports Python 3.10, 3.11, 3.12, and 3.13.

Check out the :ref:`installation <installation>` section for information on how 
to install pybear. 

See the :doc:`usage` section for ideas about when and where to use pybear.

See the :doc:`api` section for further information on the available modules.

View on `GitHub <https://github.com/PylarBear/pybear>`_


Mission Statement
=================
pybear seeks to add to and/or enhance existing functionality in the Python data analytics 
ecosystem.

pybear runs on all actively supported versions of Python.

pybear must seamlessly integrate into conventional Python data analytics workflows.
The way pybear does this is by using the scikit-learn API.

Every pybear estimator/transformer seeks to fulfill 4 objectives beyond basic functionality:

1) handle missing data: robust handling of all nan-like values (numpy.nan, pandas.NA, etc.)

2) fast processing with parallelism: use joblib when there is benefit

3) bigger than memory data: every module is capable of incremental learning

4) accept all common containers: accepts numpy, pandas, polars, and scipy sparse containers. 


.. toctree::
   :maxdepth: 2
   :caption: Contents:


Contents
--------

.. toctree::
   :titlesonly:

   installation
   usage
   api
   dev
   CONTRIBUTING
