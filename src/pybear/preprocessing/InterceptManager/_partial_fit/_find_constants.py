# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from .._type_aliases import (
    DataType
)
from numbers import Integral, Real

import numpy as np
from joblib import Parallel

from ._column_getter import _column_getter
from ._parallel_constant_finder import _parallel_constant_finder
from ._merge_constants import _merge_constants





def _find_constants(
    _X: DataType,
    _constant_columns: dict[int, any],
    _equal_nan: bool,
    _rtol: Real,
    _atol: Real,
    _n_jobs: Integral
) -> dict[int, any]:


    """
    Pizza make a recipe here.
    
    
    
    
    Parameters
    ----------
    _X:
        DataType -
    _equal_nan:
        bool -
    _rtol:
        numbers.Real -
    _atol:
        numbers.Real -
    _n_jobs:
        numbers.Integral -
    
    
    Return
    ------
    -
        _new_constants: dict[int, any] - dictionary of the indices of the
            columns of constants and the values in them.
    
    
    """





    # out is list[Union[Literal[False], any]]
    # the idxs of the list match the idxs of the data
    args = (_equal_nan, _rtol, _atol)
    joblib_kwargs = {
        'prefer': 'processes', 'return_as': 'list', 'n_jobs': _n_jobs
    }
    out = Parallel(**joblib_kwargs)(_parallel_constant_finder(
        _column_getter(_X, _col_idx), *args) for _col_idx in range(_X.shape[1])
    )


    # convert 'out' to dict[idx, value] of only the columns of constants
    # do the mask the long way, the constants could be zeros
    MASK = np.fromiter((v is False for v in out), dtype=bool)
    values = out[MASK]
    idxs = np.arange(len(out))[MASK]

    _new_constants = dict((zip(idxs, values)))

    del MASK, idxs, values

    # merge newly found constant columns with those found during
    # previous partial fits
    _constant_columns_ = _merge_constants(_constant_columns, _new_constants)
















