# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from .._type_aliases import DataType
from typing_extensions import Union
import numpy.typing as npt

from ._column_getter import _column_getter

import numpy as np

from joblib import Parallel, delayed



def _find_duplicates(
    _X:DataType,
    _n_jobs: Union[int, None]
) -> list[list[int]]:

    """
    Find identical columns in X. Create a list of lists, where each list
    indicates the zero-based column indices of columns that are identical.
    For example, if column indices 0 and 23 are identical, and indices 8,
    12, and 19 are identical, the returned object would be
    [[0, 23], [8, 12, 19]].


    Parameters
    ----------
    _X:
        DataType - The data to be deduplicated.


    Return
    ------
    -
        GROUPS: list[list[int]] - lists indicating the column indices of
            identical columns.


    """


    duplicates_: dict[int: list[int]] = {i: [] for i in range(_X.shape[1])}

    _all_duplicates = []  # not used later, just a helper to track duplicates


    def _column_comparer(
        column1: npt.NDArray[any],
        column2: npt.NDArray[any]
    ):
        return np.array_equal(column1, column2)

    kwargs = {'return_as':'list', 'prefer':'processes', 'n_jobs':_n_jobs}

    # .shape works for np, pd, and scipy.sparse
    for col_idx1 in range(_X.shape[1] - 2):

        if col_idx1 in _all_duplicates:
            continue

        # PIZZA COME BACK AND PUT JOBLIB
        RANGE = range(col_idx1 + 1, _X.shape[1])
        IDXS = [i for i in RANGE if i not in _all_duplicates]

        hits = Parallel(**kwargs)(
            delayed(_column_comparer)(
                *_column_getter(_X, col_idx1, col_idx2)) for col_idx2 in IDXS
        )

        if any(hits):
            _all_duplicates.append(col_idx1)

        for idx, hit in zip(IDXS, hits):
            if hit:
                duplicates_[col_idx1].append(idx)
                _all_duplicates.append(idx)

        """
        24_10_03_08_24_00 code that worked * * * * * * * * *
        # .shape works for np, pd, and scipy.sparse
        for col_idx2 in range(col_idx1 + 1, _X.shape[1]):

            if col_idx2 in _all_duplicates:
                continue

            if _column_comparer(_X, col_idx1, col_idx2):
                duplicates_[col_idx1].append(col_idx2)
                _all_duplicates.append(col_idx1)
                _all_duplicates.append(col_idx2)
        """

    del _all_duplicates, kwargs, RANGE, IDXS, hits

    # ONLY RETAIN INFO FOR COLUMNS THAT ARE DUPLICATE
    duplicates_ = {k: v for k, v in duplicates_.items() if len(v) > 0}

    # UNITE DUPLICATES INTO GROUPS
    GROUPS = []
    for idx1, v1 in duplicates_.items():
        __ = sorted([idx1] + v1)
        GROUPS.append(__)

    return GROUPS



