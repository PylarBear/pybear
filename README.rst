pybear
======

|PyPI Build Status|

.. |PyPI Build Status| image:: https://github.com/PylarBear/pybear/actions/workflows/pypi-publish.yml/badge.svg
   :target: https://github.com/PylarBear/pybear/actions/workflows/pypi-publish.yml

|TestPyPI Build Status|

.. |TestPyPI Build Status| image:: https://github.com/PylarBear/pybear/actions/workflows/testpypi-publish.yml/badge.svg
   :target: https://github.com/PylarBear/pybear/actions/workflows/testpypi-publish.yml

|Test Status 313|

.. |Test Status 313| image:: https://github.com/PylarBear/pybear/actions/workflows/matrix-tests-py313.yml/badge.svg
   :target: https://github.com/PylarBear/pybear/actions/workflows/matrix-tests-py313.yml

|Test Status 312|

.. |Test Status 312| image:: https://github.com/PylarBear/pybear/actions/workflows/matrix-tests-py312.yml/badge.svg
   :target: https://github.com/PylarBear/pybear/actions/workflows/matrix-tests-py312.yml

|Test Status 311|

.. |Test Status 311| image:: https://github.com/PylarBear/pybear/actions/workflows/matrix-tests-py311.yml/badge.svg
   :target: https://github.com/PylarBear/pybear/actions/workflows/matrix-tests-py311.yml

|Test Status 310|

.. |Test Status 310| image:: https://github.com/PylarBear/pybear/actions/workflows/matrix-tests-py310.yml/badge.svg
   :target: https://github.com/PylarBear/pybear/actions/workflows/matrix-tests-py310.yml

|Test Status 39|

.. |Test Status 39| image:: https://github.com/PylarBear/pybear/actions/workflows/matrix-tests-py39.yml/badge.svg
   :target: https://github.com/PylarBear/pybear/actions/workflows/matrix-tests-py39.yml

|Doc Status|

.. |Doc Status| image:: https://readthedocs.org/projects/ml/badge/?version=latest
   :target: //pybear.readthedocs.io/
   :alt: Documentation Status

|PyPI Downloads|

.. |PyPI Downloads| image:: https://img.shields.io/pypi/dm/pybear.svg?label=PyPI%20downloads
   :target: https://pypi.org/project/pybear/

|Version Status|

.. |Version Status| image:: https://img.shields.io/pypi/v/pybear.svg
   :target: https://pypi.python.org/pypi/pybear/

|PyPi|

.. |PyPi| image:: https://img.shields.io/pypi/v/pybear
   :target: https://pypi.org/project/pybear

.. _documentation: https://pybear.readthedocs.io/

.. |PythonVersion| replace:: >=3.9, <3.14
.. |DaskVersion| replace:: >=X.X.X
.. |DaskMLVersion| replace:: >=X.X.X
.. |DistributedVersion| replace:: >=X.X.X
.. |JoblibVersion| replace:: >=X.X.X
.. |NumpyVersion| replace:: >=1.26.0
.. |PandasVersion| replace:: >=X.X.X
.. |ScikitLearnVersion| replace:: >=1.1.3
.. |TypingExtensionsVersion| replace:: >=X.X.X



Cool, but not frozen, packages to augment your Python data analytics experience.

pybear is a Python computing library that augments data analytics functionality 
found in the popular numpy, scikit-learn, dask, and dask_ml libraries.

See documentation_ for more information.

Website: https://github.com/PylarBear/pybear

License
-------

BSD 3-Clause License. See `License File <https://github.com/PylarBear/pybear/blob/main/LICENSE>`__.

=======

Installation
------------

Dependencies
~~~~~~~~~~~~

pybear requires:

- Python (|PythonVersion|)
- dask (|DaskVersion|)
- dask_ml (|DaskMLVersion|)
- distributed (|DistributedVersion|)
- joblib (|JoblibVersion|)
- numpy (|NumPyVersion|)
- pandas (|PandasVersion|)
- scikit-learn (|ScikitLearnVersion|)
- typing_extensions (|TypingExtensionsVersion|)

pybear 0.1 is tested via GitHub Actions to run on Linux, Windows, and 
MacOS, with Python versions 3.9, 3.10, 3.11, and 3.12. pybear is not 
tested on earlier versions, but some features may work.

User installation
~~~~~~~~~~~~~~~~~

The only way to install pybear is from PyPI using ``pip``::

    pip install pybear

Conda distributions are not available at this time.

=======

Major Modules
-------------

AutoGridSearchCV
~~~~~~~~~~~~~~~~
Perform multiple uninterrupted passes of grid search with sci-kit learn 
GridSearchCV utilizing progressively narrower search grids. Access via 
pybear.model_selection.AutoGridSearchCV.

AutoGridSearchCVDask
~~~~~~~~~~~~~~~~~~~~
Perform multiple uninterrupted passes of grid search with dask_ml GridSearchCV 
and dask objects utilizing progressively narrower search grids. Access via 
pybear.model_selection.AutoGridSearchCVDask.

autogridsearch_wrapper
~~~~~~~~~~~~~~~~~~~~~~
Create your own auto-gridsearch class. A function that wraps any scikit-learn 
or dask_ml GridSearchCV module to create an identical GridSearch class that 
performs multiple passes of grid search using progressively narrower search 
grids. Access via pybear.model_selection.autogridsearch_wrapper.

GSTCV (GridSearchThresholdCV)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Perform conventional grid search on a classifier with concurrent threshold 
search. Finds the global optima for the passed parameters and thresholds. 
Fully compliant with the scikit-learn GridSearchCV API. Access via 
pybear.model_selection.GSTCV.

GSTCVDask (GridSearchThresholdCV for Dask)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Perform conventional grid search on a classifier with concurrent threshold 
search using dask objects in parallel and distributed environments. Finds the 
global optima for the passed parameters and thresholds. Fully compliant with 
the dask_ml/scikit-learn GridSearchCV API. Access via 
pybear.model_selection.GSTCVDask.

AutoGSTCV
~~~~~~~~~
Perform multiple uninterrupted passes of grid search with pybear GSTCV 
utilizing progressively narrower search grids. Access via 
pybear.model_selection.AutoGSTCV.

AutoGSTCVDask
~~~~~~~~~~~~~
Perform multiple uninterrupted passes of grid search with pybear GSTCVDask 
utilizing progressively narrower search grids. Access via 
pybear.model_selection.AutoGSTCVDask.

MinCountTransformer
~~~~~~~~~~~~~~~~~~~
Perform minimum frequency thresholding on numerical or categorical data 
simultaneously across an entire array of data. Violates the scikit-learn API 
in that datasets are modified along the example axis (examples may be deleted.)
Otherwise is fully compliant with the sci-kit learn transformer API, with fit, 
transform, and partial_fit methods. Access via 
pybear.preprocessing.MinCountTransformer.

ColumnDeduplicateTransformer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Identify and selectively remove duplicate columns in numerical or categorical 
data. Fully compliant with the scikit-learn transformer API, with fit, 
transform, and partial_fit methods. Perfect for removing duplicate columns from 
one-hot encoded data in a scikit-learn pipeline. Also fits and transforms data 
batch-wise, such as with dask-ml Incremental and ParallelPostFit wrappers.
Access via pybear.preprocessing.ColumnDeduplicateTransformer.

InterceptManager
~~~~~~~~~~~~~~~~
A scikit-style transformer that identifies and manages constant columns in a 
dataset. IM can remove all, selectively keep one, or append a column of 
constants. Handles numerical & non-numerical data, and nan-like values. Does 
batch-wise fitting via a partial_fit method, and can be wrapped with dask_ml 
Incremental and ParallelPostFit wrappers. Access via 
pybear.preprocessing.InterceptManager.

SlimPolyFeatures
~~~~~~~~~~~~~~~~
Perform a polynomial feature expansion on a dataset omitting constant and 
duplicate columns. Follows the standard scikit-learn transformer API. Handles 
scipy sparse matrices/arrays. Suitable for sklearn pipelines. Has a partial_fit 
method for batch-wise training and can be wrapped with dask_ml Incremental and
ParallelPostFit wrappers. Access via pybear.preprocessing.SlimPolyFeatures.

Lexicon
~~~~~~~
A class object that has an attribute that is a list of the words in the English 
lexicon, as well as other methods and attributes that have summary statistics 
about the words. Currently has over 68,000 words, but is not 100% exhaustive 
though attempts have been made to do so. Users are able to modify the contents  
of the lexicon on their local installation via built-in methods.
Access via pybear.feature_extraction.text.Lexicon.

TextStatistics
~~~~~~~~~~~~~~
A scikit-like class that ingests 1D vectors of strings and compiles statistics 
about the characters and strings seen during fitting. Has a partial_fit method 
that allows it to accumulate information for multiple batches of data. Reported 
information includes number of strings seen, average length of strings, maximum 
and minimum length, character frequency, starts-with frequency, and others.
Access via pybear.feature_extraction.text.TextStatistics.

=======

Changelog
---------

See the `changelog <https://github.com/PylarBear/pybear/blob/main/CHANGELOG.md>`__
for a history of notable changes to pybear.

=======

Development
-----------

Important links
~~~~~~~~~~~~~~~

- Official source code repo: https://github.com/PylarBear/pybear
- Download releases: https://pypi.org/project/pybear/
- Issue tracker: https://github.com/PylarBear/pybear/issues

Source code
~~~~~~~~~~~

You can clone the latest source code with the command::

    git clone https://github.com/PylarBear/pybear.git

Contributing
~~~~~~~~~~~~

pybear is not ready for contributions at this time!

Testing
~~~~~~~

After installation, you can launch the test suite from outside the pybear
root directory (you will need to have pytest installed in your environment)::

    pytest pybear

Project History
---------------

The project originated in the early 2020's as a collection of miscellaneous 
private modules to enhance the scikit-learn and dask_ml ecosystem. In 2024, the 
modules were formalized and bundled together for their first release as pybear.

Help and Support
----------------

Documentation
~~~~~~~~~~~~~

- HTML documentation: https://pybear.readthedocs.io/

Communication
~~~~~~~~~~~~~

- GitHub Discussions: https://github.com/PylarBear/pybear/discussions
- Website: https://github.com/PylarBear/pybear





