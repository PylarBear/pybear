PyBear
======

|Build Status| |Doc Status| |Version Status| |PyPi|

.. |Build Status| image:: https://github.com/PylarBear/pybear/actions/workflows/python-publish.yml/badge.svg
   :target: https://github.com/PylarBear/pybear/actions/workflows/python-publish.yml
.. |Doc Status| image:: https://readthedocs.org/projects/ml/badge/?version=latest
   :target: //pybear.readthedocs.io/
   :alt: Documentation Status
.. |Discourse| image:: https://img.shields.io/discourse/users?logo=discourse&server=https%3A%2F%2Fpybear.discourse.group
   :alt: Discuss pybear-related things and ask for help
   :target: https://github.com/PylarBear/pybear/discussions
.. |Version Status| image:: https://img.shields.io/pypi/v/pybear.svg
   :target: https://pypi.python.org/pypi/pybear/
.. |PyPi| image:: https://img.shields.io/pypi/v/pybear
   :target: https://pypi.org/project/pybear

.. _documentation: https://pybear.readthedocs.io/

.. |PythonMinVersion| replace:: 3.9
.. |DaskMinVersion| replace:: X.X.X
.. |DaskMLMinVersion| replace:: X.X.X
.. |DistributedMinVersion| replace:: X.X.X
.. |JoblibMinVersion| replace:: X.X.X
.. |NumpyMinVersion| replace:: X.X.X
.. |PandasMinVersion| replace:: X.X.X
.. |ScikitLearnMinVersion| replace:: X.X.X




Cool, but not frozen, packages to augment your Python data analytics experience.

pybear is a python computing library for data analytics meant to augment 
functionality found in the popular numpy, scikit-learn, dask, and dask_ml libraries.

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

- Python (>= |PythonMinVersion|)
- dask (>= |DaskMinVersion|)
- dask_ml (>= |DaskMLMinVersion|)
- distributed (>= |DistributedMinVersion|)
- joblib (>= |JoblibMinVersion|)
- numpy (>= |NumPyMinVersion|)
- pandas (>= |PandasMinVersion|)
- scikit-learn (>= |ScikitLearnMinVersion|)

pybear 0.1 requires Python 3.9 or newer.
pybear is not tested on earlier versions, but some features may work.


User installation
~~~~~~~~~~~~~~~~~

The only way to install PyBear is from PyPI using ``pip``::

    pip install pybear

Conda distributions are not available at this time.

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

PyBear is not ready for contributions at this time!

Testing
~~~~~~~

After installation, you can launch the test suite from outside the source
directory (you will need to have pytest installed)::

    pytest pybear

Project History
---------------

The project originated in the early 2020's as a collection of miscelleanous 
private modules to enhance the scikit-learn and dask_ml ecosystem. In 2024, 
the modules were formalized and bundled together for their first release as 
pybear.

Help and Support
----------------

Documentation
~~~~~~~~~~~~~

- HTML documentation: https://pybear.readthedocs.io/

Communication
~~~~~~~~~~~~~

- GitHub Discussions: https://github.com/PylarBear/pybear/discussions
- Website: https://github.com/PylarBear/pybear





