# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

from ..autogridsearch.autogridsearch_wrapper import autogridsearch_wrapper
from ..autogridsearch import autogridsearch_docs

from dask_ml.model_selection import GridSearchCV


# def another_wrapper_for_docstrings():
#
#     DaskAutoGridSearch = type(
#         'DaskAutoGridSearch',
#         (autogridsearch_wrapper(GridSearchCV),),
#         {'__doc__': autogridsearch_docs,
#          '__init__.__doc__': autogridsearch_docs}
#     )
#
#     DaskAutoGridSearch.__init__.__doc__ = autogridsearch_docs
#
#     return DaskAutoGridSearch
#
#
#
# DaskAutoGridSearch = another_wrapper_for_docstrings()



DaskAutoGridSearch = autogridsearch_wrapper(GridSearchCV)










