# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from .._type_aliases import InternalDataContainer
from typing_extensions import Union

from numbers import Real

import itertools

import numpy as np
import pandas as pd
import polars as pl
import scipy.sparse as ss
import joblib

from ._columns_getter import _columns_getter
from ._parallel_column_comparer import _parallel_column_comparer





def _find_duplicates(
    _X: InternalDataContainer,
    _rtol: Real,
    _atol: Real,
    _equal_nan: bool,
    _n_jobs: Union[int, None]
) -> list[list[int]]:

    """
    Find identical columns in X. Create a list of lists, where each list
    indicates the zero-based column indices of columns that are identical.
    For example, if column indices 0 and 23 are identical, and indices 8,
    12, and 19 are identical, the returned object would be
    [[0, 23], [8, 12, 19]]. It is important that the first indices of
    each subset be sorted ascending in the outer container, i.e., in
    this example, 0 is before 8.


    Parameters
    ----------
    _X:
        {array-like, scipy sparse matrix} of shape (n_samples,
        n_features) - The data to be deduplicated. _X must indexable,
        therefore scipy sparse coo, dia, and bsr are prohibited. There
        is no conditioning of the data here, it must be passed to this
        module in suitable state.
    _rtol:
        numbers.Real - the relative difference tolerance for equality.
        Must be a non-boolean, non-negative, real number. See
        numpy.allclose.
    _atol:
        numbers.Real - the absolute difference tolerance for equality.
        Must be a non-boolean, non-negative, real number. See
        numpy.allclose.
    _equal_nan:
        bool, default = False - When comparing pairs of columns row by
        row:
        If equal_nan is True, exclude from comparison any rows where one
        or both of the values is/are nan. If one value is nan, this
        essentially assumes that the nan value would otherwise be the
        same as its non-nan counterpart. When both are nan, this
        considers the nans as equal (contrary to the default numpy
        handling of nan, where numpy.nan != numpy.nan) and will not in
        and of itself cause a pair of columns to be marked as unequal.
        If equal_nan is False and either one or both of the values in
        the compared pair of values is/are nan, consider the pair to be
        not equivalent, thus making the column pair not equal. This is
        in line with the normal numpy handling of nan values.
    n_jobs:
        Union[int, None], default = -1 - The number of joblib Parallel
        jobs to use when comparing columns. The default is to use
        processes, but can be overridden externally using a joblib
        parallel_config context manager. The default number of jobs is
        -1 (all processors).


    Return
    ------
    -
        GROUPS: list[list[int]] - lists indicating the column indices of
            identical columns.


    """

    # validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    assert (isinstance(_X, (np.ndarray, (pd.core.frame.DataFrame, pl.DataFrame)))
            or hasattr(_X, 'toarray'))

    assert not isinstance(_X,
        (ss.coo_matrix, ss.coo_array, ss.dia_matrix,
         ss.dia_array, ss.bsr_matrix, ss.bsr_array)
    )

    assert isinstance(_rtol, Real) and _rtol >= 0
    assert isinstance(_atol, Real) and _atol >= 0
    assert isinstance(_equal_nan, bool)
    assert isinstance(_n_jobs, (int, type(None)))
    # END validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    # pizza it may not be necessary to pre-fill this thing... be we are appending below
    duplicates_: dict[int, list[int]] = {int(i): [] for i in range(_X.shape[1])}

    _all_duplicates = []  # not used later, just a helper to track duplicates

    args = (_rtol, _atol, _equal_nan)


    _n_cols = 200


    for col_idx1 in range(_X.shape[1] - 1):

        if col_idx1 in _all_duplicates:
            continue

        _column1 = _columns_getter(_X, col_idx1)

        # make a list of col_idx2's
        RANGE = range(col_idx1 + 1, _X.shape[1])
        IDXS = [i for i in RANGE if i not in _all_duplicates]

        if len(IDXS) < 2 * _n_cols:
            hits = []
            for col_idx2 in IDXS:
                hits.append(_parallel_column_comparer(
                    _column1, _columns_getter(_X, int(col_idx2)), *args
                )[0])
        else:
            with joblib.parallel_config(prefer='processes', n_jobs=_n_jobs):
                hits = joblib.Parallel(return_as='list')(
                    joblib.delayed(_parallel_column_comparer)(
                        _column1,
                        _columns_getter(
                            _X,
                            tuple(map(int, np.array(IDXS)[range(i, min(i + _n_cols, len(IDXS)))]))
                        ),
                        *args
                    ) for i in range(0, len(IDXS), _n_cols)
                )

            hits = list(itertools.chain(*hits))

        if any(hits):
            _all_duplicates.append(col_idx1)

            for idx, hit in zip(IDXS, hits):
                if hit:
                    duplicates_[col_idx1].append(idx)
                    _all_duplicates.append(idx)

        """
        # code that worked pre-joblib * * * * * * * * *
        for col_idx2 in range(col_idx1 + 1, _X.shape[1]):

            if col_idx2 in _all_duplicates:
                continue

            _column2 = _columns_getter(_X, col_idx2)

            if _parallel_column_comparer( _column1, _column2, *args):
                duplicates_[col_idx1].append(col_idx2)
                _all_duplicates.append(col_idx1)
                _all_duplicates.append(col_idx2)
        """

    del _all_duplicates, RANGE, IDXS, col_idx1, _column1, hits

    # ONLY RETAIN INFO FOR COLUMNS THAT ARE DUPLICATE
    duplicates_ = {int(k): v for k, v in duplicates_.items() if len(v) > 0}

    # UNITE DUPLICATES INTO GROUPS
    GROUPS = []
    for idx1, v1 in duplicates_.items():
        __ = sorted([int(idx1)] + v1)
        GROUPS.append(__)

    # ALL SETS OF DUPLICATES MUST HAVE AT LEAST 2 ENTRIES
    for _set in GROUPS:
        assert len(_set) >= 2


    return GROUPS









