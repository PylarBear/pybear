.. pybear documentation master file, created by
   sphinx-quickstart on Mon Jun 30 09:26:37 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the pybear documentation
===================================

pybear is a scikit-learn-style Python computing library that augments data 
analytics functionality found in popular packages like scikit-learn and xgboost.

Check out the :ref:`installation <installation>` section for information on how 
to install pybear. See the :doc:`usage` section for further information on the 
available modules.

.. note::

   This project is under active development.

Mission Statement
=================
pybear must have seamless integration into typical Python data analytics 
workflows, therefore it uses the scikit-learn API. All pybear transformers have 
fit and transform methods.

Additionally, every pybear module seeks to deliver on 4 principles:

1) handle missing data: robust handling of all nan-like values (numpy.nan, pandas.NA, etc.)

2) fast processing with parallelism: use joblib when there is benefit

3) bigger than memory data: every module has a partial_fit method for incremental learning

4) accept all popular containers: accepts numpy, pandas, polars, and scipy sparse containers. 


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
   build
