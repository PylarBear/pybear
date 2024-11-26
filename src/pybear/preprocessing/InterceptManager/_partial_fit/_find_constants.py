# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from .._type_aliases import DataFormatType
from typing_extensions import Union

from numbers import Integral, Real
import uuid

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ._column_getter import _column_getter
from ._parallel_constant_finder import _parallel_constant_finder
from ._merge_constants import _merge_constants





def _find_constants(
    _X: DataFormatType,
    _old_constant_columns: dict[int, any],
    _equal_nan: bool,
    _rtol: Real,
    _atol: Real,
    _n_jobs: Union[Integral, None]
) -> dict[int, any]:


    """
    Scan across the columns of _X looking for columns of constants.
    Use _column_getter() to pull columns from _X as numpy vectors.
    Use _parallel_constant_finder() to determine the constancy of a
    column with respect to rtol, atol, and equal_nan. Render the constant
    columns found as dict[int, any], with column indices as keys and
    constants as values. Use _merge_constants() to combine constants
    found in the current partial fit with those found in previous
    partial fits.

    
    Parameters
    ----------
    _X:
        {array-like, scipy sparse} of shape (n_samples, n_features) -
        The data for which to identify constant columns.
    _old_constant_columns:
        dict[int, any] - constant column indices and their values found
        in previous partial fits.
    _equal_nan:
        bool - If equal_nan is True, exclude nan-likes from computations
        that discover constant columns. This essentially assumes that
        the nan value would otherwise be equal to the mean of the non-nan
        values in the same column. If equal_nan is False and any value
        in a column is nan, do not assume that the nan value is equal to
        the mean of the non-nan values in the same column, thus making
        the column non-constant. This is in line with the normal numpy
        handling of nan values.
    _rtol:
        numbers.Real - The relative difference tolerance for equality.
        Must be a non-boolean, non-negative, real number. See
        numpy.allclose.
    _atol:
        numbers.Real - The absolute difference tolerance for equality.
        Must be a non-boolean, non-negative, real number. See
        numpy.allclose.
    _n_jobs:
        Union[numbers.Integral, None] - The number of joblib Parallel
        jobs to use when scanning the data for columns of constants. The
        default is to use processes, but can be overridden externally
        using a joblib parallel_config context manager. The default
        number of jobs is -1 (all processors). To get maximum speed
        benefit, pybear recommends using the default setting.
    
    
    Return
    ------
    -
        _new_constants: dict[int, any] - dictionary of the indices of the
            columns of constants and the values in them.
    
    
    """



    assert isinstance(_X, (np.ndarray, pd.core.frame.DataFrame)) \
           or hasattr(_X, 'toarray')
    assert isinstance(_old_constant_columns, dict)
    if len(_old_constant_columns):
        assert max(_old_constant_columns) < _X.shape[1]
    assert isinstance(_equal_nan, bool)
    try:
        float(_rtol)
        float(_atol)
    except:
        raise AssertionError
    assert isinstance(_n_jobs, (int, type(None)))
    if _n_jobs:
        assert _n_jobs >= -1


    # out is list[Union[uuid.uuid4, any]]
    # the idxs of the list match the idxs of the data

    args = (_equal_nan, _rtol, _atol)
    joblib_kwargs = {
        'prefer': 'processes', 'return_as': 'list', 'n_jobs': _n_jobs
    }
    out = Parallel(**joblib_kwargs)(delayed(_parallel_constant_finder)(
        _column_getter(_X, _col_idx), *args) for _col_idx in range(_X.shape[1])
    )

    # convert 'out' to dict[idx, value] for only the columns of constants
    _new_constants = {}
    for idx, v in enumerate(out):
        # do this out the long way, to do vectorization everything needs
        # to be converted to np, and the np dtypes mess up the dict keys.
        # _parallel_constant_finder() returns the constant value when
        # the column is constant, otherwise returns a uuid4 identifier.
        if isinstance(v, uuid.UUID):
            continue
        else:
            _new_constants[idx] = v


    # merge newly found constant columns with those found during
    # previous partial fits
    return _merge_constants(_old_constant_columns, _new_constants, _rtol, _atol)
















