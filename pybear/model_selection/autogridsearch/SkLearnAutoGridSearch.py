# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

from ..autogridsearch.autogridsearch_wrapper import autogridsearch_wrapper
from ..autogridsearch import autogridsearch_docs

from sklearn.model_selection import GridSearchCV


# def another_wrapper_for_docstrings():
#
#     SklearnAutoGridSearch = type(
#         'SklearnAutoGridSearch',
#         (autogridsearch_wrapper(GridSearchCV),),
#         {'__doc__': autogridsearch_docs,
#          '__init__.__doc__': autogridsearch_docs}
#     )
#
#     SklearnAutoGridSearch.__init__.__doc__ = autogridsearch_docs
#
#     return SklearnAutoGridSearch
#
#
#
# SklearnAutoGridSearch = another_wrapper_for_docstrings()




SklearnAutoGridSearch = autogridsearch_wrapper(GridSearchCV)




