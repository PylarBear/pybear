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
import pandas as pd
from joblib import Parallel, delayed

from ._column_getter import _column_getter
from ._parallel_constant_finder import _parallel_constant_finder
from ._merge_constants import _merge_constants





def _find_constants(
    _X: DataType,
    _old_constant_columns: dict[int, any],
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
        DataType - pizza!
    _old_constant_columns:
        dict[int, any] - constant column indices and their values found
        in previous partial fits.
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



    assert isinstance(_X, (np.ndarray, pd.core.frame.DataFrame)) or hasattr(_X, 'toarray')
    assert isinstance(_old_constant_columns, dict)
    assert max(_old_constant_columns) < _X.shape[1] if len(_old_constant_columns) else True
    assert isinstance(_equal_nan, bool)
    assert isinstance(_rtol, (int, float))
    assert isinstance(_atol, (int, float))
    assert isinstance(_n_jobs, (int, type(None)))



    # out is list[Union[Literal[False], any]]
    # the idxs of the list match the idxs of the data
    args = (_equal_nan, _rtol, _atol)
    joblib_kwargs = {
        'prefer': 'processes', 'return_as': 'list', 'n_jobs': _n_jobs
    }
    out = Parallel(**joblib_kwargs)(delayed(_parallel_constant_finder)(
        _column_getter(_X, _col_idx), *args) for _col_idx in range(_X.shape[1])
    )


    # convert 'out' to dict[idx, value] of only the columns of constants
    _new_constants = {}
    for idx, v in enumerate(out):
        # do this out the long way, to do vectorization everything needs
        # to be converted to np, and the np dtypes mess up the dict keys
        if v is not False:  # the constants could be zeros, use is
            _new_constants[idx] = v


    # merge newly found constant columns with those found during
    # previous partial fits
    return _merge_constants(_old_constant_columns, _new_constants, _rtol, _atol)
















