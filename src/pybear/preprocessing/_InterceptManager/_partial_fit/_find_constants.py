# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing_extensions import (
    Any,
    Union
)
from .._type_aliases import (
    InternalDataContainer,
    ConstantColumnsType
)

import itertools
import numbers
import uuid

import numpy as np
import pandas as pd
import polars as pl
import scipy.sparse as ss
import joblib

from ._columns_getter import _columns_getter
from ._parallel_constant_finder import _parallel_constant_finder
from ._merge_constants import _merge_constants



def _find_constants(
    _X: InternalDataContainer,
    _old_constant_columns: Union[ConstantColumnsType, None],
    _equal_nan: bool,
    _rtol: numbers.Real,
    _atol: numbers.Real,
    _n_jobs: Union[numbers.Integral, None]
) -> ConstantColumnsType:

    """
    Scan across the columns of _X looking for columns of constants.
    Use _columns_getter() to pull columns from _X as numpy vectors.
    Use _parallel_constant_finder() to determine the constancy of a
    column with respect to rtol, atol, and equal_nan. Render the constant
    columns found as dict[int, Any], with column indices as keys and
    constants as values. Use _merge_constants() to combine constants
    found in the current partial fit with those found in previous
    partial fits.

    
    Parameters
    ----------
    _X:
        array-like of shape (n_samples, n_features) - The data to
        be searched for constant columns. _X will be passed to
        _columns_getter and must observe the restrictions imposed
        there. _X should be in the correct state when passed to this
        module.
    _old_constant_columns:
        Union[ConstantColumnsType, None] - constant column indices and
        their values found in previous partial fits.
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
        jobs to use when scanning the data for columns of constants.
    
    
    Return
    ------
    -
        _new_constants: ConstantColumnsType - dictionary of the indices
        of the columns of constants and the values in them for the
        current partial fit.
    
    
    """

    # validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **
    assert isinstance(_X,
        (np.ndarray, pd.core.frame.DataFrame, pl.DataFrame, ss.csc_array,
         ss.csc_matrix)
    )

    assert isinstance(_old_constant_columns, (dict, type(None)))
    if _old_constant_columns and len(_old_constant_columns):
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
    # END validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **


    # out is list[Union[uuid.uuid4, Any]]
    # the idxs of the list match the idxs of the data

    # number of columns to send in a job to joblib
    _n_cols = 200

    args = (_equal_nan, _rtol, _atol)

    if _X.shape[1] < _n_cols:
        # if X columns < chunk columns just run it under a for loop
        out = []
        for _c_idx in range(_X.shape[1]):
            out += _parallel_constant_finder(_columns_getter(_X, _c_idx), *args)
    else:
        _new_cols = _n_cols * (_X.shape[1]//_n_cols + 1)
        with joblib.parallel_config(prefer='processes', n_jobs=_n_jobs):
            out = joblib.Parallel(return_as='list')(
                joblib.delayed(_parallel_constant_finder)(
                    _columns_getter(
                        _X,
                        tuple(range(i, min(i + _n_cols, _X.shape[1])))
                    ),
                    *args
                ) for i in range(0, _new_cols, _n_cols)
            )

        out = list(itertools.chain(*out))


    # convert 'out' to dict[idx, value] for only the columns of constants
    _new_constants = {}
    for idx, v in enumerate(out):
        # do this out the long way, to do vectorization everything needs
        # to be converted to np, and the np dtypes mess up the dict keys.
        # _parallel_constant_finder() returns the constant value when
        # the column is constant, otherwise returns a uuid4 identifier.
        if not isinstance(v, uuid.UUID):
            _new_constants[idx] = v


    # merge newly found constant columns with those found during
    # previous partial fits
    return _merge_constants(_old_constant_columns, _new_constants, _rtol, _atol)




