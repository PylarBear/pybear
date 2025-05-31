# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing_extensions import Union
from .._type_aliases import (
    InternalXContainer,
    DuplicatesType
)

from collections import defaultdict
import math
import numbers
import itertools

import numpy as np
import pandas as pd
import polars as pl
import scipy.sparse as ss
import joblib

from ._columns_getter import _columns_getter
from ._parallel_chunk_comparer import _parallel_chunk_comparer

from ....utilities._union_find import union_find



def _find_duplicates(
    _X: InternalXContainer,
    _rtol: numbers.Real,
    _atol: numbers.Real,
    _equal_nan: bool,
    _n_jobs: Union[numbers.Integral, None]
) -> DuplicatesType:

    """
    Find identical columns in X. Create a list of lists, where each list
    indicates the zero-based column indices of columns that are identical.
    For example, if column indices 0 and 23 are identical, and indices
    8, 12, and 19 are identical, the returned object would be
    [[0, 23], [8, 12, 19]]. It is important that the first indices of
    each subset be sorted ascending in the outer container, i.e., in
    this example, 0 is before 8.


    Parameters
    ----------
    _X:
        array-like of shape (n_samples, n_features) - The data to be
        deduplicated. _X must indexable, therefore scipy sparse coo,
        dia, and bsr are prohibited. There is no conditioning of the
        data here, it must be passed to this module in suitable state.
    _rtol:
        numbers.Real - the relative difference tolerance for equality.
        Must be a non-boolean, non-negative, real number. See
        numpy.allclose.
    _atol:
        numbers.Real - the absolute difference tolerance for equality.
        Must be a non-boolean, non-negative, real number. See
        numpy.allclose.
    _equal_nan:
        bool - When comparing pairs of columns row by row:

        If equal_nan is True, exclude from comparison any rows where one
        or both of the values is/are nan. If one value is nan, this
        essentially assumes that the nan value would otherwise be the
        same as its non-nan counterpart. When both are nan, this
        considers the nans as equal (contrary to the default numpy
        handling of nan, where numpy.nan != numpy.nan) and will not in
        and of itself cause a pair of columns to be considered unequal.
        If equal_nan is False and either one or both of the values in
        the compared pair of values is/are nan, consider the pair to be
        not equivalent, thus making the column pair not equal. This is
        in line with the normal numpy handling of nan values.
    n_jobs:
        Union[numbers.Integral, None] - The number of joblib Parallel
        jobs to use when comparing columns. The default is to use
        processes, but can be overridden externally using a joblib
        parallel_config context manager. The default number of jobs is
        -1 (all processors).


    Return
    ------
    -
        GROUPS: DuplicatesType - lists indicating the column indices of
        identical columns.

    """


    # validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    assert isinstance(_X, (np.ndarray, pd.core.frame.DataFrame, pl.DataFrame,
        ss.csc_array, ss.csc_matrix))
    assert isinstance(_rtol, numbers.Real) and _rtol >= 0
    assert isinstance(_atol, numbers.Real) and _atol >= 0
    assert isinstance(_equal_nan, bool)
    assert isinstance(_n_jobs, (int, type(None)))
    # END validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    args = (_rtol, _atol, _equal_nan)

    # set the batch size for joblib
    _n_cols = 20

    if _X.shape[1] < 2 * _n_cols:

        _all_duplicates = []  # not used later, just a helper to track duplicates

        _dupls = []
        for col_idx1 in range(_X.shape[1] - 1):

            if col_idx1 in _all_duplicates:
                continue

            _column1 = _columns_getter(_X, col_idx1)

            # make a list of col_idx2's
            RANGE = range(col_idx1 + 1, _X.shape[1])
            IDXS = [i for i in RANGE if i not in _all_duplicates]

            for col_idx2 in IDXS:

                _match = _parallel_chunk_comparer(
                    _columns_getter(_X, col_idx1),
                    (col_idx1, ),
                    _columns_getter(_X, col_idx2),
                    (col_idx2, ),
                    *args
                )

                if any(_match):
                    _dupls += _match
                    if col_idx1 not in _all_duplicates:
                        _all_duplicates.append(col_idx1)
                    if col_idx2 not in _all_duplicates:
                        _all_duplicates.append(col_idx2)

        del _all_duplicates, RANGE, IDXS

    else:
        with joblib.parallel_config(prefer='processes', n_jobs=_n_jobs):
            _dupls = joblib.Parallel(return_as='list')(
                joblib.delayed(_parallel_chunk_comparer)(
                    _columns_getter(_X, tuple(range(i * _n_cols, min(_n_cols * (i+1), _X.shape[1])))),
                    tuple(range(i * _n_cols, min(_n_cols * (i + 1), _X.shape[1]))),
                    _columns_getter(_X, tuple(range(j * _n_cols, min(_n_cols * (j+1), _X.shape[1])))),
                    tuple(range(j * _n_cols, min(_n_cols * (j + 1), _X.shape[1]))),
                    *args
                    ) for i, j in itertools.combinations_with_replacement(range(math.ceil(_X.shape[1]/_n_cols)), 2)
                )

        _dupls = list(itertools.chain(*_dupls))
        assert all(map(isinstance, _dupls, (tuple for i in _dupls)))

    # v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^

    duplicates_ = union_find(_dupls)

    # Sort each component and the final result for consistency
    duplicates_ = [sorted(component) for component in duplicates_]
    duplicates_ = sorted(duplicates_, key=lambda x: x[0])


    # ALL SETS OF DUPLICATES MUST HAVE AT LEAST 2 ENTRIES
    for _set in duplicates_:
        assert len(_set) >= 2


    return duplicates_





