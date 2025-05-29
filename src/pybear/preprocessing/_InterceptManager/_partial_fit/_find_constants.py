# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from .._type_aliases import (
    InternalXContainer,
    ConstantColumnsType
)

import itertools
import numbers
import uuid

import numpy as np
import pandas as pd
import polars as pl
import scipy.sparse as ss

from ._columns_getter import _columns_getter
from ._parallel_constant_finder import _parallel_constant_finder



def _find_constants(
    _X: InternalXContainer,
    _equal_nan: bool,
    _rtol: numbers.Real,
    _atol: numbers.Real
) -> ConstantColumnsType:

    # pizza need to update this...
    # _merge_constants was taken out and put into IM.partial_fit
    # _old_constant_columns was taken out of signature
    # current X contants now returned instead of merged previous/current

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
        _columns_getter and must observe the container restrictions
        imposed there. _X should be in the correct state when passed to
        this module.
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
    
    
    Return
    ------
    -
        _new_constants: ConstantColumnsType - dictionary of the indices
        of the columns of constants and the values in them across all
        partial fits.

    """


    # validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    assert isinstance(_X,
        (np.ndarray, pd.core.frame.DataFrame, pl.DataFrame, ss.csc_array,
         ss.csc_matrix)
    )

    assert isinstance(_equal_nan, bool)
    try:
        float(_rtol)
        float(_atol)
    except:
        raise AssertionError
    # END validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    # as of 25_05_29 no longer using joblib. even with sending chunks of
    # X instead of single columns across joblib it still wasnt a
    # benefit. The cost of serializing the data is not worth it for the
    # light task of checking if the column is constant.

    # the original attempt at this was passing all col indices in _X to
    # columns_getter and passing the whole thing as np to _pcf. It turned
    # out that this was creating a brief but huge spike in RAM because
    # a full copy of the entire _X was being made. so trade off by
    # pulling smaller chunks of _X and passing to _pcf... this gives
    # the benefit of less calls to _columns_getter & _pcf than would be
    # if pulling one column at a time, still with some memory spike but
    # much smaller, and get some economy of scale with the speed of
    # scanning a ndarray chunk.
    # number of columns to pull and scan in one pass of the :for: loop
    _n_cols = 10
    args = (_equal_nan, _rtol, _atol)
    out = []
    for i in range(0, _X.shape[1], _n_cols):
        out.append(_parallel_constant_finder(
            _columns_getter(
                _X,
                tuple(range(i, min(i + _n_cols, _X.shape[1])))
            ),
            *args
        ))

    out = list(itertools.chain(*out))

    # out is list[Union[uuid.uuid4, Any]]
    # the idxs of the list match the idxs of the data

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
    return _new_constants




