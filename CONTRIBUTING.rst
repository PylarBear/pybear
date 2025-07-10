Contributing To pybear
======================

pybear is not actively seeking contributions. However, at the same time, pybear 
does not want to turn away good work that enhances pybear and the Python data 
analytics ecosystem. Below is the development framework for pybear from its 
inception, and must continue with any future contributions. In addition to the 
below guidelines, it is best to use existing pybear source code as a reference 
to guide you as to what is expected in contributions. Code should be internally 
consistent with the design and conventions of existing pybear modules.

This project adheres to a Code of Conduct. When you contribute to pybear, or 
engage with the pybear community, you are expected to adhere to these rules.

`Code of Conduct <https://github.com/PylarBear/pybear/blob/main/CODE_OF_CONDUCT.md>`__


The Original Pybear Mission Statement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pybear seeks to add to and/or enhance existing functionality in the Python data 
analytics ecosystem.

pybear runs on all actively supported versions of Python.

pybear must seamlessly integrate into conventional Python data analytics workflows.
The way pybear does this is by using the scikit-learn API.

Every pybear module seeks to fulfill 4 objectives:

1. handle missing data: robust handling of all nan-like values (numpy.nan, pandas.NA, etc.)
2. fast processing with parallelism: use joblib when there is benefit
3. bigger than memory data: every module has a partial_fit method for incremental learning
4. accept all common containers: accepts numpy, pandas, polars, and scipy sparse containers

Going forward, contributions must uphold the original mission statement. 


Functional Code
~~~~~~~~~~~~~~~

All modules must follow the appropriate scikit-learn API for their type 
(e.g. transformer, estimator).

All major modules must always accept numpy ndarrays, pandas dataframes, and
polars dataframes. They must also accept all scipy sparse matrix/array formats
(as of the time of this writing there are seven) when the data is strictly
numeric. Some modules may also accept Python lists, tuples, and sets, if there
is a good reason (consider ragged arrays in pybear text analytics, which uses
Python built-ins.) pybear generally encourages the use of memory-optimized
containers over Python built-ins except in the case of text analytics. Any
containers beyond the the ones listed here, especially if they require
importing a new package, should be avoided unless there is a compelling case
for the addition.

All modules must robustly handle any nan-like values that could be found in the
containers listed above (e.g, numpy.nan, pandas.NA). pybear recommends using
the pybear.utilities.nan_mask module and/or its variants. Also see the
documentation in these 'nan_mask' modules for the full discussion on what
nan-like values are handled by pybear.

If a module can be written in such a way that joblib can demonstrably improve
speed over linear code, then the module must be written in that way and use joblib.

All modules must have a partial_fit method if it is technically possible. If it 
can have it, it must have it.


Code Formatting
~~~~~~~~~~~~~~~

pybear follows `PEP8 <https://peps.python.org/pep-0008/>`__ conventions. While no
specific format linter is required (e.g. flake8), contributions should follow 
the basic PEP8 guidelines. Use 4 spaces for indents. Use a right margin of 72 
characters for docstrings and in-line comments. For code, use a right margin 
of 79, with allowance for overflow up to 88 characters when it "makes sense." 
Use 2 blank lines before and after function definitions. These are just a few 
of the PEP8 formatting recommendations, see the actual spec for the full details. 
There is latitude afforded by PEP8, but if unsure, use existing pybear code as 
a formatting reference. The formatting of new code should not be conspicuously
different from existing pybear code.


Docs
~~~~

pybear uses the numpydocs standard for docstrings. Please refer to the numpydoc 
`Style Guide <https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard>`__.
All docstrings must thoroughly document the purpose and functionality of their 
respective modules. Functions must at least have the "Parameters", "Returns" and
"Examples" sections. Classes must at least have the "Parameters", "Attributes", 
"Returns" and "Examples" sections. Sections like "Notes", "References", "Raises", 
"See Also", etc., are optional, but encouraged if they add clarity. Type hints 
are expected, and must be accurate and consistent with the source code. See the 
`Type Hints`_ section for continuation of docstring guidance.

There is a dual mandate that docs must render accurately and aesthetically in
PyCharm tooltips and on the `pybear website <https://pybear.readthedocs.io/en/stable/>`__.
pybear uses sphinx with the numpydoc extension to automatically render docstrings
and publish them to Read The Docs. Unfortunately, there is not a one-to-one
relationship between the PyCharm linters and sphinx, meaning, that formatting
that displays perfectly with sphinx is not-so-perfect for PyCharm, and vice-versa.
pybear seeks to optimize sphinx first (i.e., follow the numpydoc standard exactly),
but try to reasonably accommodate the PyCharm linters when possible.

These guidelines must be followed for every module and submodule, whether public 
or private.


Type Hints
~~~~~~~~~~

Type hints are required in the code body and docstring in every module, public 
or private. In public modules, they greatly improve clarity for the user. In 
all cases, they greatly improve clarity during the development and maintenance 
of the source code. Type hints are expected to be accurate and consistent. 
Type hinting can be subjective. When in doubt, refer to existing pybear source 
code for examples of usage.


Test
~~~~

pybear uses pytest. If you want to contribute to the public API then you must 
submit tests with it. Every module must be tested comprehensively for center 
cases and edge cases. If the module takes various types of data containers, 
all of them must be tested. Test for correct handling of invalid inputs, not 
just valid inputs. pybear uses pytest-cov to calculate coverage and does not 
use # pragma: no cover. At first release test coverage was 93% and going forward 
this number should stay over 90%. 


How To Submit
~~~~~~~~~~~~~

pybear uses GitHub to manage versions. See the
`pybear homepage <https://github.com/PylarBear/pybear>`__ on GitHub. Fork the
pybear repo or clone pybear to your local device. Use internet resources (such 
as GitHub’s documentation or community tutorials) for details about how to get 
the source code from GitHub. Follow the above guidelines and make your changes. 
Create a pull request, explain what changes you have made in detail, and ask 
for a review.






