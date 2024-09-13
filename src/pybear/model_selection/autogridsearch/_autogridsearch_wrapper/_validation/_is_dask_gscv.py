# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import sys



def _is_dask_gscv(_GridSearchParent) -> bool:

    """

    Validate that the parent grid search class is or is not a dask_ml
    grid search. Must pass the class, not an instance. Look at the path
    of the class's module, and if the term 'dask' is in it, assume this
    is a dask grid search.


    Parameters
    ----------
    _GridSearchParent:
        A grid search class, not an instance. The parent of the auto-
        gridsearch instance.


    Return
    ------
    -
        _is_dask_gscv:
            bool - whether or not the parent grid search is a dask grid
            search.


    """


    try:
        _module = sys.modules[_GridSearchParent.__module__].__file__
    except:
        raise AttributeError(f"could not access module path for "
            f"'{_GridSearchParent.__name__}' while validating is / is not "
            f"a dask classifier")

    return 'dask_ml' in str(_module).lower()




